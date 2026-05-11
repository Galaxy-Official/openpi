# LeRobot Dataset Inventory

- Root: `/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/umipolicy/lerobot/src/Data`
- Generated: 2026-05-11 19:51:56
- Video probe mode: `count_frames`
- Latent inspect mode: `sample`

## Dataset Directory Format

```text
{root}/
  {task_name}/
    meta/
      info.json                  # declared features, fps, total episodes/frames
      stats.json                 # feature statistics when available
      tasks.parquet              # task_index -> natural-language task
      episodes/chunk-XXX/file-YYY.parquet
                                  # episode_index, length, data/video file indices
    data/chunk-XXX/file-YYY.parquet
                                  # frame-level tabular data: state, action, forces, timestamps
    videos/{video_key}/[chunk-XXX/]file-YYY.mp4
                                  # video modalities, e.g. wrist image and tactile streams
    latents/chunk-XXX/{feature_key}/episode_{episode}_{start}_{end}.pth
                                  # optional latent features, usually aligned by frame span
```

## Task Summary

| task | info_total_episodes | meta_episode_rows | info_total_frames | data_parquet_rows | vision_wrist | tactile_left | tactile_right | force_left | force_right | latent_wrist | touch_force_fusion | vision_touch_force_contrast | vision_latent_contrast | total_human | warnings |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 429_erase_board_lerobot | 215 | 215 | 31342 | 31342 | 31342 | 31342 | 31342 | 31342 | 31342 | 0 | 31342 | 31342 | 0 | 1.9 GiB | 0 |
| 430_clamp_seal_lerobot | 230 | 230 | 34746 | 34746 | 34746 | 34746 | 34746 | 34746 | 34746 | 34746 | 34746 | 34746 | 34746 | 3.7 GiB | 0 |
| 430_towel_hanging_lerobot | 207 | 207 | 33310 | 33310 | 33310 | 33310 | 33310 | 33310 | 33310 | 33310 | 33310 | 33310 | 33310 | 3.3 GiB | 0 |
| 501_bread_moving_lerobot | 199 | 199 | 20518 | 20518 | 20518 | 20518 | 20518 | 20518 | 20518 | 0 | 20518 | 20518 | 0 | 1.3 GiB | 0 |
| 505_screw_lerobot | 203 | 203 | 44155 | 44155 | 44155 | 44155 | 44155 | 44155 | 44155 | 0 | 44155 | 44155 | 0 | 3.0 GiB | 0 |
| 505_stiring_lerobot | 147 | 147 | 27380 | 27380 | 27380 | 27380 | 27380 | 27380 | 27380 | 0 | 27380 | 27380 | 0 | 1.9 GiB | 0 |
| 506_open_bottle_lerobot | 162 | 162 | 29023 | 29023 | 29023 | 29023 | 29023 | 29023 | 29023 | 0 | 29023 | 29023 | 0 | 2.0 GiB | 0 |
| 506_peg_flowers_lerobot | 212 | 212 | 29591 | 29591 | 29591 | 29591 | 29591 | 29591 | 29591 | 0 | 29591 | 29591 | 0 | 2.2 GiB | 0 |

## Modality Totals Across Tasks

| modality_group | feature | source | task_count | files | frames_or_rows | bytes_human |
| --- | --- | --- | --- | --- | --- | --- |
| action | action | data_parquet | 8 | 39 | 250065 | 9.1 MiB |
| depth_image | observation.images.phone_depth | data_parquet | 8 | 39 | 250065 | 3.4 GiB |
| force | observation.forces.left | data_parquet | 8 | 39 | 250065 | 292.6 KiB |
| force | observation.forces.right | data_parquet | 8 | 39 | 250065 | 569.3 KiB |
| index | episode_index | data_parquet | 8 | 39 | 250065 | 151.9 KiB |
| index | frame_index | data_parquet | 8 | 39 | 250065 | 1.4 MiB |
| index | index | data_parquet | 8 | 39 | 250065 | 1.4 MiB |
| index | task_index | data_parquet | 8 | 39 | 250065 | 151.9 KiB |
| index | timestamp | data_parquet | 8 | 39 | 250065 | 1.3 MiB |
| latent | observation.images.wrist | latent_files | 2 | 437 | 68056 | 2.2 GiB |
| pose | end_pose | data_parquet | 8 | 39 | 250065 | 228.0 KiB |
| pose | start_pose | data_parquet | 8 | 39 | 250065 | 228.0 KiB |
| state | observation.state | data_parquet | 8 | 39 | 250065 | 9.1 MiB |
| state | observation.state_phone | data_parquet | 8 | 39 | 250065 | 7.7 MiB |
| tactile_video | observation.tactiles.left | video_files | 8 | 8 | 250065 | 688.3 MiB |
| tactile_video | observation.tactiles.right | video_files | 8 | 8 | 250065 | 570.8 MiB |
| vision_video | observation.images.phone | video_files | 8 | 36 | 250065 | 6.4 GiB |
| vision_video | observation.images.wrist | video_files | 8 | 35 | 250065 | 5.9 GiB |

## Declared Feature Schema

| feature | modality_group | dtypes | shapes | task_count |
| --- | --- | --- | --- | --- |
| action | action | ['float32'] | [[10]] | 8 |
| end_pose | pose | ['float32'] | [[6]] | 8 |
| episode_index | index | ['int64'] | [[1]] | 8 |
| frame_index | index | ['int64'] | [[1]] | 8 |
| index | index | ['int64'] | [[1]] | 8 |
| observation.forces.left | force | ['float32'] | [[1]] | 8 |
| observation.forces.right | force | ['float32'] | [[1]] | 8 |
| observation.images.phone | vision_video | ['video'] | [[480, 640, 3]] | 8 |
| observation.images.phone_depth | depth_image | ['image'] | [[480, 640, 1]] | 8 |
| observation.images.wrist | vision_video | ['video'] | [[480, 640, 3]] | 8 |
| observation.state | state | ['float32'] | [[10]] | 8 |
| observation.state_phone | state | ['float32'] | [[6]] | 8 |
| observation.tactiles.left | tactile_video | ['video'] | [[224, 224, 3]] | 8 |
| observation.tactiles.right | tactile_video | ['video'] | [[224, 224, 3]] | 8 |
| start_pose | pose | ['float32'] | [[6]] | 8 |
| task_index | index | ['int64'] | [[1]] | 8 |
| timestamp | index | ['float32'] | [[1]] | 8 |

## Per-Task Modality Detail

| task | modality_group | feature | source | files | frames_or_rows | count_source | bytes_human |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 429_erase_board_lerobot | action | action | data_parquet | 5 | 31342 | parquet_rows | 1.1 MiB |
| 429_erase_board_lerobot | pose | end_pose | data_parquet | 5 | 31342 | parquet_rows | 30.6 KiB |
| 429_erase_board_lerobot | index | episode_index | data_parquet | 5 | 31342 | parquet_rows | 20.7 KiB |
| 429_erase_board_lerobot | index | frame_index | data_parquet | 5 | 31342 | parquet_rows | 175.5 KiB |
| 429_erase_board_lerobot | index | index | data_parquet | 5 | 31342 | parquet_rows | 176.2 KiB |
| 429_erase_board_lerobot | force | observation.forces.left | data_parquet | 5 | 31342 | parquet_rows | 16.5 KiB |
| 429_erase_board_lerobot | force | observation.forces.right | data_parquet | 5 | 31342 | parquet_rows | 76.0 KiB |
| 429_erase_board_lerobot | depth_image | observation.images.phone_depth | data_parquet | 5 | 31342 | parquet_rows | 469.2 MiB |
| 429_erase_board_lerobot | state | observation.state | data_parquet | 5 | 31342 | parquet_rows | 1.1 MiB |
| 429_erase_board_lerobot | state | observation.state_phone | data_parquet | 5 | 31342 | parquet_rows | 986.8 KiB |
| 429_erase_board_lerobot | pose | start_pose | data_parquet | 5 | 31342 | parquet_rows | 30.6 KiB |
| 429_erase_board_lerobot | index | task_index | data_parquet | 5 | 31342 | parquet_rows | 20.7 KiB |
| 429_erase_board_lerobot | index | timestamp | data_parquet | 5 | 31342 | parquet_rows | 170.9 KiB |
| 429_erase_board_lerobot | vision_video | observation.images.phone | video_files | 4 | 31342 | ffprobe_count_frames | 739.0 MiB |
| 429_erase_board_lerobot | vision_video | observation.images.wrist | video_files | 3 | 31342 | ffprobe_count_frames | 522.5 MiB |
| 429_erase_board_lerobot | tactile_video | observation.tactiles.left | video_files | 1 | 31342 | ffprobe_count_frames | 86.8 MiB |
| 429_erase_board_lerobot | tactile_video | observation.tactiles.right | video_files | 1 | 31342 | ffprobe_count_frames | 73.2 MiB |
| 430_clamp_seal_lerobot | action | action | data_parquet | 5 | 34746 | parquet_rows | 1.3 MiB |
| 430_clamp_seal_lerobot | pose | end_pose | data_parquet | 5 | 34746 | parquet_rows | 33.1 KiB |
| 430_clamp_seal_lerobot | index | episode_index | data_parquet | 5 | 34746 | parquet_rows | 22.2 KiB |
| 430_clamp_seal_lerobot | index | frame_index | data_parquet | 5 | 34746 | parquet_rows | 193.7 KiB |
| 430_clamp_seal_lerobot | index | index | data_parquet | 5 | 34746 | parquet_rows | 194.4 KiB |
| 430_clamp_seal_lerobot | force | observation.forces.left | data_parquet | 5 | 34746 | parquet_rows | 17.7 KiB |
| 430_clamp_seal_lerobot | force | observation.forces.right | data_parquet | 5 | 34746 | parquet_rows | 68.2 KiB |
| 430_clamp_seal_lerobot | depth_image | observation.images.phone_depth | data_parquet | 5 | 34746 | parquet_rows | 493.5 MiB |
| 430_clamp_seal_lerobot | state | observation.state | data_parquet | 5 | 34746 | parquet_rows | 1.3 MiB |
| 430_clamp_seal_lerobot | state | observation.state_phone | data_parquet | 5 | 34746 | parquet_rows | 1.1 MiB |
| 430_clamp_seal_lerobot | pose | start_pose | data_parquet | 5 | 34746 | parquet_rows | 33.1 KiB |
| 430_clamp_seal_lerobot | index | task_index | data_parquet | 5 | 34746 | parquet_rows | 22.2 KiB |
| 430_clamp_seal_lerobot | index | timestamp | data_parquet | 5 | 34746 | parquet_rows | 188.7 KiB |
| 430_clamp_seal_lerobot | vision_video | observation.images.phone | video_files | 5 | 34746 | ffprobe_count_frames | 910.9 MiB |
| 430_clamp_seal_lerobot | vision_video | observation.images.wrist | video_files | 5 | 34746 | ffprobe_count_frames | 972.8 MiB |
| 430_clamp_seal_lerobot | tactile_video | observation.tactiles.left | video_files | 1 | 34746 | ffprobe_count_frames | 95.9 MiB |
| 430_clamp_seal_lerobot | tactile_video | observation.tactiles.right | video_files | 1 | 34746 | ffprobe_count_frames | 78.0 MiB |
| 430_clamp_seal_lerobot | latent | observation.images.wrist | latent_files | 230 | 34746 | filename_span_end_minus_start | 1.1 GiB |
| 430_towel_hanging_lerobot | action | action | data_parquet | 5 | 33310 | parquet_rows | 1.2 MiB |
| 430_towel_hanging_lerobot | pose | end_pose | data_parquet | 5 | 33310 | parquet_rows | 30.2 KiB |
| 430_towel_hanging_lerobot | index | episode_index | data_parquet | 5 | 33310 | parquet_rows | 20.0 KiB |
| 430_towel_hanging_lerobot | index | frame_index | data_parquet | 5 | 33310 | parquet_rows | 184.5 KiB |
| 430_towel_hanging_lerobot | index | index | data_parquet | 5 | 33310 | parquet_rows | 185.2 KiB |
| 430_towel_hanging_lerobot | force | observation.forces.left | data_parquet | 5 | 33310 | parquet_rows | 15.9 KiB |
| 430_towel_hanging_lerobot | force | observation.forces.right | data_parquet | 5 | 33310 | parquet_rows | 57.5 KiB |
| 430_towel_hanging_lerobot | depth_image | observation.images.phone_depth | data_parquet | 5 | 33310 | parquet_rows | 397.1 MiB |
| 430_towel_hanging_lerobot | state | observation.state | data_parquet | 5 | 33310 | parquet_rows | 1.2 MiB |
| 430_towel_hanging_lerobot | state | observation.state_phone | data_parquet | 5 | 33310 | parquet_rows | 1.0 MiB |
| 430_towel_hanging_lerobot | pose | start_pose | data_parquet | 5 | 33310 | parquet_rows | 30.2 KiB |
| 430_towel_hanging_lerobot | index | task_index | data_parquet | 5 | 33310 | parquet_rows | 20.0 KiB |
| 430_towel_hanging_lerobot | index | timestamp | data_parquet | 5 | 33310 | parquet_rows | 180.0 KiB |
| 430_towel_hanging_lerobot | vision_video | observation.images.phone | video_files | 5 | 33310 | ffprobe_count_frames | 922.7 MiB |
| 430_towel_hanging_lerobot | vision_video | observation.images.wrist | video_files | 5 | 33310 | ffprobe_count_frames | 831.0 MiB |
| 430_towel_hanging_lerobot | tactile_video | observation.tactiles.left | video_files | 1 | 33310 | ffprobe_count_frames | 90.0 MiB |
| 430_towel_hanging_lerobot | tactile_video | observation.tactiles.right | video_files | 1 | 33310 | ffprobe_count_frames | 76.5 MiB |
| 430_towel_hanging_lerobot | latent | observation.images.wrist | latent_files | 207 | 33310 | filename_span_end_minus_start | 1.0 GiB |
| 501_bread_moving_lerobot | action | action | data_parquet | 3 | 20518 | parquet_rows | 770.4 KiB |
| 501_bread_moving_lerobot | pose | end_pose | data_parquet | 3 | 20518 | parquet_rows | 27.3 KiB |
| 501_bread_moving_lerobot | index | episode_index | data_parquet | 3 | 20518 | parquet_rows | 19.2 KiB |
| 501_bread_moving_lerobot | index | frame_index | data_parquet | 3 | 20518 | parquet_rows | 118.7 KiB |
| 501_bread_moving_lerobot | index | index | data_parquet | 3 | 20518 | parquet_rows | 119.3 KiB |
| 501_bread_moving_lerobot | force | observation.forces.left | data_parquet | 3 | 20518 | parquet_rows | 39.9 KiB |
| 501_bread_moving_lerobot | force | observation.forces.right | data_parquet | 3 | 20518 | parquet_rows | 38.5 KiB |
| 501_bread_moving_lerobot | depth_image | observation.images.phone_depth | data_parquet | 3 | 20518 | parquet_rows | 285.4 MiB |
| 501_bread_moving_lerobot | state | observation.state | data_parquet | 3 | 20518 | parquet_rows | 766.2 KiB |
| 501_bread_moving_lerobot | state | observation.state_phone | data_parquet | 3 | 20518 | parquet_rows | 645.5 KiB |
| 501_bread_moving_lerobot | pose | start_pose | data_parquet | 3 | 20518 | parquet_rows | 27.3 KiB |
| 501_bread_moving_lerobot | index | task_index | data_parquet | 3 | 20518 | parquet_rows | 19.2 KiB |
| 501_bread_moving_lerobot | index | timestamp | data_parquet | 3 | 20518 | parquet_rows | 114.4 KiB |
| 501_bread_moving_lerobot | vision_video | observation.images.phone | video_files | 3 | 20518 | ffprobe_count_frames | 538.6 MiB |
| 501_bread_moving_lerobot | vision_video | observation.images.wrist | video_files | 3 | 20518 | ffprobe_count_frames | 405.5 MiB |
| 501_bread_moving_lerobot | tactile_video | observation.tactiles.left | video_files | 1 | 20518 | ffprobe_count_frames | 57.8 MiB |
| 501_bread_moving_lerobot | tactile_video | observation.tactiles.right | video_files | 1 | 20518 | ffprobe_count_frames | 49.1 MiB |
| 505_screw_lerobot | action | action | data_parquet | 6 | 44155 | parquet_rows | 1.6 MiB |
| 505_screw_lerobot | pose | end_pose | data_parquet | 6 | 44155 | parquet_rows | 31.1 KiB |
| 505_screw_lerobot | index | episode_index | data_parquet | 6 | 44155 | parquet_rows | 19.6 KiB |
| 505_screw_lerobot | index | frame_index | data_parquet | 6 | 44155 | parquet_rows | 238.9 KiB |
| 505_screw_lerobot | index | index | data_parquet | 6 | 44155 | parquet_rows | 239.6 KiB |
| 505_screw_lerobot | force | observation.forces.left | data_parquet | 6 | 44155 | parquet_rows | 87.1 KiB |
| 505_screw_lerobot | force | observation.forces.right | data_parquet | 6 | 44155 | parquet_rows | 115.8 KiB |
| 505_screw_lerobot | depth_image | observation.images.phone_depth | data_parquet | 6 | 44155 | parquet_rows | 576.0 MiB |
| 505_screw_lerobot | state | observation.state | data_parquet | 6 | 44155 | parquet_rows | 1.6 MiB |
| 505_screw_lerobot | state | observation.state_phone | data_parquet | 6 | 44155 | parquet_rows | 1.4 MiB |
| 505_screw_lerobot | pose | start_pose | data_parquet | 6 | 44155 | parquet_rows | 31.1 KiB |
| 505_screw_lerobot | index | task_index | data_parquet | 6 | 44155 | parquet_rows | 19.6 KiB |
| 505_screw_lerobot | index | timestamp | data_parquet | 6 | 44155 | parquet_rows | 234.5 KiB |
| 505_screw_lerobot | vision_video | observation.images.phone | video_files | 6 | 44155 | ffprobe_count_frames | 1.1 GiB |
| 505_screw_lerobot | vision_video | observation.images.wrist | video_files | 6 | 44155 | ffprobe_count_frames | 1.1 GiB |
| 505_screw_lerobot | tactile_video | observation.tactiles.left | video_files | 1 | 44155 | ffprobe_count_frames | 123.1 MiB |
| 505_screw_lerobot | tactile_video | observation.tactiles.right | video_files | 1 | 44155 | ffprobe_count_frames | 99.6 MiB |
| 505_stiring_lerobot | action | action | data_parquet | 5 | 27380 | parquet_rows | 1019.7 KiB |
| 505_stiring_lerobot | pose | end_pose | data_parquet | 5 | 27380 | parquet_rows | 21.8 KiB |
| 505_stiring_lerobot | index | episode_index | data_parquet | 5 | 27380 | parquet_rows | 14.2 KiB |
| 505_stiring_lerobot | index | frame_index | data_parquet | 5 | 27380 | parquet_rows | 150.0 KiB |
| 505_stiring_lerobot | index | index | data_parquet | 5 | 27380 | parquet_rows | 150.5 KiB |
| 505_stiring_lerobot | force | observation.forces.left | data_parquet | 5 | 27380 | parquet_rows | 51.5 KiB |
| 505_stiring_lerobot | force | observation.forces.right | data_parquet | 5 | 27380 | parquet_rows | 110.4 KiB |
| 505_stiring_lerobot | depth_image | observation.images.phone_depth | data_parquet | 5 | 27380 | parquet_rows | 419.8 MiB |
| 505_stiring_lerobot | state | observation.state | data_parquet | 5 | 27380 | parquet_rows | 1017.2 KiB |
| 505_stiring_lerobot | state | observation.state_phone | data_parquet | 5 | 27380 | parquet_rows | 870.0 KiB |
| 505_stiring_lerobot | pose | start_pose | data_parquet | 5 | 27380 | parquet_rows | 21.8 KiB |
| 505_stiring_lerobot | index | task_index | data_parquet | 5 | 27380 | parquet_rows | 14.2 KiB |
| 505_stiring_lerobot | index | timestamp | data_parquet | 5 | 27380 | parquet_rows | 146.8 KiB |
| 505_stiring_lerobot | vision_video | observation.images.phone | video_files | 4 | 27380 | ffprobe_count_frames | 714.2 MiB |
| 505_stiring_lerobot | vision_video | observation.images.wrist | video_files | 4 | 27380 | ffprobe_count_frames | 683.1 MiB |
| 505_stiring_lerobot | tactile_video | observation.tactiles.left | video_files | 1 | 27380 | ffprobe_count_frames | 70.7 MiB |
| 505_stiring_lerobot | tactile_video | observation.tactiles.right | video_files | 1 | 27380 | ffprobe_count_frames | 60.9 MiB |
| 506_open_bottle_lerobot | action | action | data_parquet | 5 | 29023 | parquet_rows | 1.1 MiB |
| 506_open_bottle_lerobot | pose | end_pose | data_parquet | 5 | 29023 | parquet_rows | 23.9 KiB |
| 506_open_bottle_lerobot | index | episode_index | data_parquet | 5 | 29023 | parquet_rows | 15.6 KiB |
| 506_open_bottle_lerobot | index | frame_index | data_parquet | 5 | 29023 | parquet_rows | 159.6 KiB |
| 506_open_bottle_lerobot | index | index | data_parquet | 5 | 29023 | parquet_rows | 160.2 KiB |
| 506_open_bottle_lerobot | force | observation.forces.left | data_parquet | 5 | 29023 | parquet_rows | 12.5 KiB |
| 506_open_bottle_lerobot | force | observation.forces.right | data_parquet | 5 | 29023 | parquet_rows | 23.0 KiB |
| 506_open_bottle_lerobot | depth_image | observation.images.phone_depth | data_parquet | 5 | 29023 | parquet_rows | 448.2 MiB |
| 506_open_bottle_lerobot | state | observation.state | data_parquet | 5 | 29023 | parquet_rows | 1.1 MiB |
| 506_open_bottle_lerobot | state | observation.state_phone | data_parquet | 5 | 29023 | parquet_rows | 919.6 KiB |
| 506_open_bottle_lerobot | pose | start_pose | data_parquet | 5 | 29023 | parquet_rows | 23.9 KiB |
| 506_open_bottle_lerobot | index | task_index | data_parquet | 5 | 29023 | parquet_rows | 15.6 KiB |
| 506_open_bottle_lerobot | index | timestamp | data_parquet | 5 | 29023 | parquet_rows | 156.2 KiB |
| 506_open_bottle_lerobot | vision_video | observation.images.phone | video_files | 4 | 29023 | ffprobe_count_frames | 749.5 MiB |
| 506_open_bottle_lerobot | vision_video | observation.images.wrist | video_files | 4 | 29023 | ffprobe_count_frames | 711.2 MiB |
| 506_open_bottle_lerobot | tactile_video | observation.tactiles.left | video_files | 1 | 29023 | ffprobe_count_frames | 81.9 MiB |
| 506_open_bottle_lerobot | tactile_video | observation.tactiles.right | video_files | 1 | 29023 | ffprobe_count_frames | 65.5 MiB |
| 506_peg_flowers_lerobot | action | action | data_parquet | 5 | 29591 | parquet_rows | 1.1 MiB |
| 506_peg_flowers_lerobot | pose | end_pose | data_parquet | 5 | 29591 | parquet_rows | 30.1 KiB |
| 506_peg_flowers_lerobot | index | episode_index | data_parquet | 5 | 29591 | parquet_rows | 20.5 KiB |
| 506_peg_flowers_lerobot | index | frame_index | data_parquet | 5 | 29591 | parquet_rows | 166.5 KiB |
| 506_peg_flowers_lerobot | index | index | data_parquet | 5 | 29591 | parquet_rows | 167.1 KiB |
| 506_peg_flowers_lerobot | force | observation.forces.left | data_parquet | 5 | 29591 | parquet_rows | 51.3 KiB |
| 506_peg_flowers_lerobot | force | observation.forces.right | data_parquet | 5 | 29591 | parquet_rows | 79.9 KiB |
| 506_peg_flowers_lerobot | depth_image | observation.images.phone_depth | data_parquet | 5 | 29591 | parquet_rows | 419.6 MiB |
| 506_peg_flowers_lerobot | state | observation.state | data_parquet | 5 | 29591 | parquet_rows | 1.1 MiB |
| 506_peg_flowers_lerobot | state | observation.state_phone | data_parquet | 5 | 29591 | parquet_rows | 927.4 KiB |
| 506_peg_flowers_lerobot | pose | start_pose | data_parquet | 5 | 29591 | parquet_rows | 30.1 KiB |
| 506_peg_flowers_lerobot | index | task_index | data_parquet | 5 | 29591 | parquet_rows | 20.5 KiB |
| 506_peg_flowers_lerobot | index | timestamp | data_parquet | 5 | 29591 | parquet_rows | 161.9 KiB |
| 506_peg_flowers_lerobot | vision_video | observation.images.phone | video_files | 5 | 29591 | ffprobe_count_frames | 845.4 MiB |
| 506_peg_flowers_lerobot | vision_video | observation.images.wrist | video_files | 5 | 29591 | ffprobe_count_frames | 840.9 MiB |
| 506_peg_flowers_lerobot | tactile_video | observation.tactiles.left | video_files | 1 | 29591 | ffprobe_count_frames | 82.2 MiB |
| 506_peg_flowers_lerobot | tactile_video | observation.tactiles.right | video_files | 1 | 29591 | ffprobe_count_frames | 67.9 MiB |

## Warnings And Consistency Checks

_No warnings._

## Output Files

- `task_summary.csv`: one row per task, useful for pretraining sampling ratios.
- `modalities_by_task.csv`: one row per task/modality/source.
- `modality_totals.csv`: totals grouped by modality across all tasks.
- `feature_schema.csv`: declared feature schema from `meta/info.json`.
- `dataset_inventory.json`: full machine-readable inventory with warnings and probe details.
