from collections import defaultdict
from functools import partial as bind
import embodied.core.vision as vision

import elements
import embodied
import numpy as np
import cv2


def eval_only(make_agent, make_env, make_logger, args):
  assert args.from_checkpoint

  agent = make_agent()
  logger = make_logger()

  logdir = elements.Path(args.logdir)
  logdir.mkdir()
  print('Logdir', logdir)
  step = logger.step
  usage = elements.Usage(**args.usage)
  agg = elements.Agg()
  epstats = elements.Agg()
  episodes = defaultdict(elements.Agg)
  should_log = elements.when.Clock(args.log_every)
  policy_fps = elements.FPS()

  @elements.timer.section('logfn')
  def logfn(tran, worker):
    episode = episodes[worker]
    tran['is_first'] and episode.reset()
    episode.add('score', tran['reward'], agg='sum')
    episode.add('length', 1, agg='sum')
    episode.add('rewards', tran['reward'], agg='stack')
    if args["task"].startswith("cr-atari"):
      episode.add('gaze_positions', tran['gaze_position'], agg='stack')
    for key, value in tran.items():
      isimage = (value.dtype == np.uint8) and (value.ndim == 3)
      if isimage and worker == 0:
        episode.add(f'policy_{key}', value, agg='stack')
      elif key.startswith('log/'):
        assert value.ndim == 0, (key, value.shape, value.dtype)
        episode.add(key + '/avg', value, agg='avg')
        episode.add(key + '/max', value, agg='max')
        episode.add(key + '/sum', value, agg='sum')
    if tran['is_last']:
      result = episode.result()
      logger.add({
          'score': result.pop('score'),
          'length': result.pop('length'),
      }, prefix='episode')

      rew = result.pop('rewards')
      if len(rew) > 1:
        result['reward_rate'] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()

      if args["task"].startswith("cr-atari"):
        vision_square_size = args["cr-atari.vision_square_size"]
        size = args["cr-atari.size"]
        gaze_heatmap_image = np.zeros(size + (1,), dtype=np.int32)
        scale_gaze_scanpath_image = 100
        gaze_scanpath_image = np.zeros((size[0]*scale_gaze_scanpath_image, size[1]*scale_gaze_scanpath_image, 1), dtype=np.uint8)
        gaze_positions = result.pop("gaze_positions")
        vision_square_count = vision.calc_vision_square_count(size=size, vision_square_size=vision_square_size)
        last_x_scaled = None
        last_y_scaled = None
        for i, gaze_position in enumerate(gaze_positions):

          x, y = vision.convert_1d_vision_square_position_to_2d_random(
              vision_square_position=gaze_position,
              vision_square_count=vision_square_count,
              vision_square_size=vision_square_size
          )
          gaze_heatmap_image[y,x] += 1

          x_scaled = x * scale_gaze_scanpath_image
          y_scaled = y * scale_gaze_scanpath_image

          if last_x_scaled is not None and last_y_scaled is not None:
              cv2.line(gaze_scanpath_image, (last_x_scaled, last_y_scaled), (x_scaled, y_scaled), (255), 1)

          cv2.putText(
              gaze_scanpath_image,
              str(i+1),
              (x_scaled + 3, y_scaled + 3),
              cv2.FONT_HERSHEY_SIMPLEX,
              0.4,
              (125),
              1,
              cv2.LINE_AA
          )
          last_x_scaled = x_scaled
          last_y_scaled = y_scaled

        # Normalize image from 0 to 1 -> scale from 0 to 255
        gaze_heatmap_image = ((gaze_heatmap_image - gaze_heatmap_image.min()) * (1/(gaze_heatmap_image.max() - gaze_heatmap_image.min()) * 255)).astype('uint8')
        logger.image("gaze_heatmap", gaze_heatmap_image)
        logger.image("gaze_scanpath", gaze_scanpath_image)
        logger.add({'gaze_positions': str(gaze_positions)})

      epstats.add(result)

  fns = [bind(make_env, i) for i in range(args.envs)]
  driver = embodied.Driver(fns, parallel=(not args.debug))
  driver.on_step(lambda tran, _: step.increment())
  driver.on_step(lambda tran, _: policy_fps.step())
  driver.on_step(logfn)

  cp = elements.Checkpoint()
  cp.agent = agent
  cp.load(args.from_checkpoint, keys=['agent'])

  print('Start evaluation')
  policy = lambda *args: agent.policy(*args, mode='eval')
  driver.reset(agent.init_policy)
  while step < args.steps:
    driver(policy, steps=10)
    if should_log(step):
      logger.add(agg.result())
      logger.add(epstats.result(), prefix='epstats')
      logger.add(usage.stats(), prefix='usage')
      logger.add({'fps/policy': policy_fps.result()})
      logger.add({'timer': elements.timer.stats()['summary']})
      logger.write()

  logger.close()
