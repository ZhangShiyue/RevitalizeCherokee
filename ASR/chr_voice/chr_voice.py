# coding=utf-8
# Copyright 2021 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Common Voice Dataset"""


import os

import datasets
from datasets.tasks import AutomaticSpeechRecognition


_DATA_URL = "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-6.1-2020-12-11/{}.tar.gz"

_CITATION = """
"""

_DESCRIPTION = """
"""

_HOMEPAGE = "https://github.com/CherokeeLanguage/cherokee-audio-data"

_LICENSE = "https://github.com/CherokeeLanguage/cherokee-audio-data/blob/main/AUDIO-LICENSE.md"

_LANGUAGES = {
    "chr": {
        "Language": "Cherokee",
        "Date": "2021-10-17",
        "Size": "X MB",
        "Version": "chr_xh_2021-10-17",
        "Validated_Hr_Total": 0,
        "Overall_Hr_Total": 0,
        "Number_Of_Voice": 0,
    },
}


class CommonVoiceConfig(datasets.BuilderConfig):
    """BuilderConfig for CommonVoice."""

    def __init__(self, name, sub_version, **kwargs):
        """
        Args:
          data_dir: `string`, the path to the folder containing the files in the
            downloaded .tar
          citation: `string`, citation for the data set
          url: `string`, url for information about the data set
          **kwargs: keyword arguments forwarded to super.
        """
        self.sub_version = sub_version
        self.language = kwargs.pop("language", None)
        self.date_of_snapshot = kwargs.pop("date", None)
        self.size = kwargs.pop("size", None)
        self.validated_hr_total = kwargs.pop("val_hrs", None)
        self.total_hr_total = kwargs.pop("total_hrs", None)
        self.num_of_voice = kwargs.pop("num_of_voice", None)
        description = f"Chr Voice speech to text dataset in {self.language} version {self.sub_version} " \
                      f"of {self.date_of_snapshot}. The dataset comprises {self.validated_hr_total} " \
                      f"of validated transcribed speech data from {self.num_of_voice} speakers. " \
                      f"The dataset has a size of {self.size}"
        super(CommonVoiceConfig, self).__init__(
            name=name, version=datasets.Version("1.0.0", ""), description=description, **kwargs
        )


class CommonVoice(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        CommonVoiceConfig(
            name=lang_id,
            language=_LANGUAGES[lang_id]["Language"],
            sub_version=_LANGUAGES[lang_id]["Version"],
            date=_LANGUAGES[lang_id]["Date"],
            size=_LANGUAGES[lang_id]["Size"],
            val_hrs=_LANGUAGES[lang_id]["Validated_Hr_Total"],
            total_hrs=_LANGUAGES[lang_id]["Overall_Hr_Total"],
            num_of_voice=_LANGUAGES[lang_id]["Number_Of_Voice"],
        )
        for lang_id in _LANGUAGES.keys()
    ]

    def _info(self):
        features = datasets.Features(
            {
                "client_id": datasets.Value("string"),
                "path": datasets.Value("string"),
                "audio": datasets.features.Audio(sampling_rate=48_000),
                "sentence": datasets.Value("string"),
                "up_votes": datasets.Value("int64"),
                "down_votes": datasets.Value("int64"),
                "age": datasets.Value("string"),
                "gender": datasets.Value("string"),
                "accent": datasets.Value("string"),
                "locale": datasets.Value("string"),
                "segment": datasets.Value("string"),
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
            task_templates=[
                AutomaticSpeechRecognition(audio_column="audio", transcription_column="sentence")
            ],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # dl_path = dl_manager.download_and_extract(_DATA_URL.format(self.config.name))
        dl_path = "/ssd-playpen/home/shiyue/RevitalizeCherokee/ASR/chr_voice"
        abs_path_to_data = os.path.join(dl_path)
        abs_path_to_clips = os.path.join(abs_path_to_data, "clips")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(abs_path_to_data, "train.tsv"),
                    "path_to_clips": abs_path_to_clips,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(abs_path_to_data, "test.tsv"),
                    "path_to_clips": abs_path_to_clips,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(abs_path_to_data, "dev.tsv"),
                    "path_to_clips": abs_path_to_clips,
                },
            ),
            datasets.SplitGenerator(
                name="other",
                gen_kwargs={
                    "filepath": os.path.join(abs_path_to_data, "other.tsv"),
                    "path_to_clips": abs_path_to_clips,
                },
            ),
            datasets.SplitGenerator(
                name="invalidated",
                gen_kwargs={
                    "filepath": os.path.join(abs_path_to_data, "invalidated.tsv"),
                    "path_to_clips": abs_path_to_clips,
                },
            ),
        ]

    def _generate_examples(self, filepath, path_to_clips):
        """Yields examples."""
        data_fields = list(self._info().features.keys())

        # audio is not a header of the csv files
        data_fields.remove("audio")
        path_idx = data_fields.index("path")

        with open(filepath, encoding="utf-8") as f:
            lines = f.readlines()
            headline = lines[0]

            column_names = headline.strip().split("\t")
            assert (
                column_names == data_fields
            ), f"The file should have {data_fields} as column names, but has {column_names}"

            for id_, line in enumerate(lines[1:]):
                field_values = line.strip().split("\t")

                # set absolute path for mp3 audio file
                field_values[path_idx] = os.path.join(path_to_clips, field_values[path_idx])

                # if data is incomplete, fill with empty values
                if len(field_values) < len(data_fields):
                    field_values += (len(data_fields) - len(field_values)) * ["''"]

                result = {key: value for key, value in zip(data_fields, field_values)}

                # set audio feature
                result["audio"] = field_values[path_idx]

                yield id_, result