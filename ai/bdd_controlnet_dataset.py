import os
import datasets
import pandas as pd

_CITATION = ""
_DESCRIPTION = ""
_HOMEPAGE = ""
_LICENSE = ""

class NewDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.1.0")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="BDD_dataset", version=VERSION, description="Image, semantic segmentation label, caption"),
    ]

    DEFAULT_CONFIG_NAME = "BDD_dataset"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION, 
            features=datasets.Features({
                "image": datasets.Image(),
                "condition": datasets.Image(),
                "caption": datasets.Value("string"),
            }),  
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # Here we assume that the path to the dataset is passed via `data_dir`
        data_dir = os.path.abspath(dl_manager.manual_dir)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": os.path.join(data_dir, "bdd_hf_dataset_train.csv"), "split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": os.path.join(data_dir, "bdd_hf_dataset_val.csv"), "split": "val"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": os.path.join(data_dir, "bdd_hf_dataset_test.csv"), "split": "test"},
            ),
        ]

    def _generate_examples(self, filepath, split):
        metadata = pd.read_csv(filepath)
        for _, row in metadata.iterrows():
            image_path = row["image"]
            image = open(image_path, "rb").read()
            conditioning_image_path = row["condition"]
            conditioning_image = open(conditioning_image_path, "rb").read()
            yield row["image"], {
                "caption": row["caption"],
                "image": {"path": image_path, "bytes": image},
                "condition": {"path": conditioning_image_path, "bytes": conditioning_image},
            }