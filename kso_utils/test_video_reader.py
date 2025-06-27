import cv2
import pytest

import kso_utils.video_reader as vr


@pytest.mark.skip(reason="we need a short example video to test this")
def test_video_reader(video_path: str, show=False):
    for use_cache in (False, True):
        reader = vr.VideoReader(path=video_path, use_cache=use_cache)

        all_frame_hashes = [
            vr._compute_fixed_hash(f.image) for f in reader.iter_frames()
        ]
        all_frame_hashes_again = [
            vr._compute_fixed_hash(f.image) for f in reader.iter_frames()
        ]
        assert all_frame_hashes == all_frame_hashes_again

        # With just the top lines:
        # Starting block 'Running with use_cache=False...
        #   EasyProfile: Running with use_cache=False took 2260.51ms
        # Starting block 'Running with use_cache=True...
        # EasyProfile: Running with use_cache=True took 1942.80ms
        count = 0
        time_snoppet_hashes = []
        for frame in reader.iter_frames(time_interval=(1, 2)):
            if show:
                cv2.imshow("frame", frame.image)
                cv2.waitKey(1)
            time_snoppet_hashes.append(vr._compute_fixed_hash(frame.image))
            count += 1
        start_ix = reader.time_to_nearest_frame(1)
        assert count > 20
        assert all_frame_hashes[start_ix : start_ix + count] == time_snoppet_hashes

        index_snippet_hashes = []

        for i, frame in enumerate(reader.iter_frames(frame_interval=(10, 20))):
            if show:
                cv2.imshow("frame", frame.image)
                cv2.waitKey(1)
            index_snippet_hashes.append(vr._compute_fixed_hash(frame.image))
            assert frame.frame_ix == 10 + i
        assert len(index_snippet_hashes) == 10
        assert all_frame_hashes[10:20] == index_snippet_hashes

        last_frame = reader.request_frame(-1)
        assert last_frame.frame_ix == reader.get_n_frames() - 1
        assert vr._compute_fixed_hash(last_frame.image) == all_frame_hashes[-1]
        assert (
            reader.request_frame(reader.get_n_frames() - 1).frame_ix
            == reader.get_n_frames() - 1
        )
        assert (
            reader.request_frame(reader.get_n_frames()).frame_ix
            == reader.get_n_frames() - 1
        )

        second_last_frame = reader.request_frame(reader.get_n_frames() - 2)
        assert second_last_frame.frame_ix == reader.get_n_frames() - 2
        assert vr._compute_fixed_hash(second_last_frame.image) == all_frame_hashes[-2]

        third_last_frame = reader.request_frame(reader.get_n_frames() - 3)
        assert third_last_frame.frame_ix == reader.get_n_frames() - 3
        assert vr._compute_fixed_hash(third_last_frame.image) == all_frame_hashes[-3]

        last_frame_again = reader.request_frame(-1)
        assert vr._compute_fixed_hash(last_frame_again.image) == all_frame_hashes[-1]

        first_frame = reader.request_frame(0)
        assert first_frame.frame_ix == 0
        assert vr._compute_fixed_hash(first_frame.image) == all_frame_hashes[0]

        random_frame = reader.request_frame(20)
        assert random_frame.frame_ix == 20
        assert vr._compute_fixed_hash(random_frame.image) == all_frame_hashes[20]
