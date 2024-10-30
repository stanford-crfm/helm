# import os
# import pytest
# from tempfile import TemporaryDirectory

# from helm.benchmark.metrics.basic_metrics import CORRECT_TAG
# from helm.benchmark.scenarios.audio_language.iemocap_audio_scenario import IEMOCAPAudioScenario
# from helm.benchmark.scenarios.scenario import Input
# from helm.common.media_object import MediaObject, MultimediaObject


# @pytest.mark.scenarios
# def test_iemocap_audio_scenario_get_instances():
#     scenario = IEMOCAPAudioScenario()
#     with TemporaryDirectory() as tmpdir:
#         actual_instances = scenario.get_instances(tmpdir)

#         assert len(actual_instances) == 5
#         expected_wav_path = os.path.join(tmpdir, "wav", "Ses01M_impro01_F000.wav")
#         assert actual_instances[0].input == Input(
#             text="",
#             multimedia_content=MultimediaObject(
#                 media_objects=[
#                     MediaObject(
#                         content_type="audio/wav",
#                         location="/tmp/tmpd414kwf1/wav/Ses01M_impro01_F000.wav")
#                 ]
#             ),
#         )
#         #         27039
#         # [0.00543212890625, 0.0048828125, 0.00537109375, 0.004608154296875, 0.005584716796875,
# 0.006622314453125, 0.0064697265625, 0.00592041015625, 0.00457763671875, 0.003387451171875]
#         assert len(actual_instances[0].references) == 1
#         assert actual_instances[0].references[0].output.text == "1.1.1"
#         assert actual_instances[0].references[0].tags == [CORRECT_TAG]
