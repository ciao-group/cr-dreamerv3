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
  SCALE_GAZE_SCANPATH_IMAGE = 100

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
        gaze_scanpath_image = np.zeros((size[0]*SCALE_GAZE_SCANPATH_IMAGE, size[1]*SCALE_GAZE_SCANPATH_IMAGE, 1), dtype=np.uint8)
        gaze_positions = result.pop("gaze_positions")
        vision_square_count = vision.calc_vision_square_count(size=size, vision_square_size=vision_square_size)
        last_x = None
        last_y = None
        for i, gaze_position in enumerate(gaze_positions):

          x, y = vision.convert_1d_vision_square_position_to_2d_random(
              vision_square_position=gaze_position,
              vision_square_count=vision_square_count,
              vision_square_size=vision_square_size
          )
          gaze_heatmap_image[y,x] += 1

          vision.add_scanpath_to_image(
              x1 = last_x,
              y1 = last_y,
              x2= x,
              y2= y,
              number = i+1,
              gaze_scanpath_image=gaze_scanpath_image,
              scale_gaze_scanpath_image=SCALE_GAZE_SCANPATH_IMAGE
          )
          last_x = x
          last_y = y

        logger.image("gaze_heatmap", vision.normalize_heatmap_image_0_to_255(gaze_heatmap_image=gaze_heatmap_image))
        logger.image("gaze_scanpath", gaze_scanpath_image)
        logger.add({'gaze_positions': np.array2string(gaze_positions,threshold=len(gaze_positions), separator=", ").replace('\n', '')})

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
