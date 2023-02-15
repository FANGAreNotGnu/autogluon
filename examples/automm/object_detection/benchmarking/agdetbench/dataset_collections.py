from base_dataset import AGDetBenchBaseDataset


class apparel_blogger_influencer(AGDetBenchBaseDataset):
    dataset_name = "apparel_blogger_influencer"


class apparel_fashion_show(AGDetBenchBaseDataset):
    dataset_name = "apparel_fashion_show"


class apparel_streetwear_daily(AGDetBenchBaseDataset):
    dataset_name = "apparel_streetwear_daily"


class chest10(AGDetBenchBaseDataset):
    dataset_name = "chest10"


class cityscapes(AGDetBenchBaseDataset):
    dataset_name = "cityscapes"


class clipart(AGDetBenchBaseDataset):
    dataset_name = "clipart"


class comic(AGDetBenchBaseDataset):
    dataset_name = "comic"


class damaged_vehicles(AGDetBenchBaseDataset):
    dataset_name = "damaged_vehicles"


class deepfruits(AGDetBenchBaseDataset):
    dataset_name = "deepfruits"


class deeplesion(AGDetBenchBaseDataset):
    dataset_name = "deeplesion"
    splits = ["train", "val", "trainval", "test"]


class dota(AGDetBenchBaseDataset):
    dataset_name = "dota"
    splits = ["train", "test", "traintest"]


class duo(AGDetBenchBaseDataset):
    dataset_name = "duo"


class ena24(AGDetBenchBaseDataset):
    dataset_name = "ena24"


class f1(AGDetBenchBaseDataset):
    dataset_name = "f1"


class kitchen(AGDetBenchBaseDataset):
    dataset_name = "kitchen"


class kitti(AGDetBenchBaseDataset):
    dataset_name = "kitti"


class lisa(AGDetBenchBaseDataset):
    dataset_name = "lisa"


class mario(AGDetBenchBaseDataset):
    dataset_name = "mario"


class minneapple(AGDetBenchBaseDataset):
    dataset_name = "minneapple"


class nfl_logo(AGDetBenchBaseDataset):
    dataset_name = "nfl_logo"


class oktoberfest(AGDetBenchBaseDataset):
    dataset_name = "oktoberfest"


class pothole(AGDetBenchBaseDataset):
    dataset_name = "pothole"
    splits = ["train", "val", "test"]


class rugrats(AGDetBenchBaseDataset):
    dataset_name = "rugrats"


class sixray(AGDetBenchBaseDataset):
    dataset_name = "sixray"


class table(AGDetBenchBaseDataset):
    dataset_name = "table"


class tt100k(AGDetBenchBaseDataset):
    dataset_name = "tt100k"


class uefa(AGDetBenchBaseDataset):
    dataset_name = "uefa"


class utensils(AGDetBenchBaseDataset):
    dataset_name = "utensils"


class vehicles_test_commercial(AGDetBenchBaseDataset):
    dataset_name = "vehicles_test_commercial"


class voc0712(AGDetBenchBaseDataset):
    dataset_name = "voc0712"
    splits = ["train", "val", "test"]


class widerface(AGDetBenchBaseDataset):
    dataset_name = "widerface"


LAUNCH = {
    "apparel_blogger_influencer": apparel_blogger_influencer,
    "apparel_fashion_show": apparel_fashion_show,
    "apparel_streetwear_daily": apparel_streetwear_daily,
    "chest10": chest10,
    "cityscapes": cityscapes,
    "clipart": clipart,
    "comic": comic,
    "damaged_vehicles": damaged_vehicles,
    "deepfruits": deepfruits,
    "deeplesion": deeplesion,
    "dota": dota,
    "duo": duo,
    "ena24": ena24,
    "f1": f1,
    "kitchen": kitchen,
    "kitti": kitti,
    "lisa": lisa,
    "mario": mario,
    "minneapple": minneapple,
    "nfl_logo": nfl_logo,
    "oktoberfest": oktoberfest,
    "pothole": pothole,
    "rugrats": rugrats,
    "sixray": sixray,
    "table": table,
    "tt100k": tt100k,
    "uefa": uefa,
    "utensils": utensils,
    "vehicles_test_commercial": vehicles_test_commercial,
    "voc0712": voc0712,
    "widerface": widerface,
}

SHORTCUT_LAUNCH = {
    "ab": apparel_blogger_influencer,  #
    "af": apparel_fashion_show,  #
    "as": apparel_streetwear_daily,  #
    "ch": chest10,  #
    "ct": cityscapes,  #
    "cl": clipart,  #
    "cm": comic,  #
    "dv": damaged_vehicles,  #
    "df": deepfruits,  #
    "dl": deeplesion,  #
    "dt": dota,  #
    "duo": duo,
    "en": ena24,  #
    "f1": f1,
    "kc": kitchen,  #
    "kt": kitti,  #
    "ls": lisa,  #
    "mr": mario,  #
    "ma": minneapple,  #
    "nfl": nfl_logo,  #
    "ok": oktoberfest,  #
    "ph": pothole,  #
    "rr": rugrats,  #
    "sr": sixray,  #
    "tb": table,  #
    "tt": tt100k,  #
    "uf": uefa,  #
    "ut": utensils,  #
    "vt": vehicles_test_commercial,  #
    "voc": voc0712,  #
    "wf": widerface,  #
}

def launch_dataset(key, data_root):
    if key in LAUNCH:
        return LAUNCH[key](data_root=data_root)
    elif key in SHORTCUT_LAUNCH:
        return SHORTCUT_LAUNCH[key](data_root=data_root)
    else:
        raise ValueError(f"Dataset Key Error: {key}")

if __name__ == "__main__":
    #for Dataset in list(LAUNCH.values())[20:]:
    #for Dataset in [LAUNCH["widerface"]]:
    #    dataset = Dataset("/media/code/datasets/AGDetBench")
    #    dataset.print_table()
    for k in LAUNCH:
        print(k)
