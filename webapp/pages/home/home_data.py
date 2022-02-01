import dill


# @cache.memoize()
def get_dataset(dataset_path):
    with open(dataset_path, "rb") as f:
        dataset = dill.load(f)

    return dataset


def set_dataset(dataset, path):
    # path_dir = pathlib.Path(UPLOAD_FOLDER_ROOT) / session_id
    # path_dir.mkdir(parents=True, exist_ok=True)
    # path = path_dir / f"{button_id}-dataset.pickle"
    with open(path, "wb") as f:
        dill.dump(dataset, f)
        dataset.df.to_csv(path.replace(".pickle", ".csv"), index=False)
    return path


# @cache.memoize()
def get_trajectory(trajectory_path):

    with open(trajectory_path, "rb") as f:
        dataset = dill.load(f)

    return dataset


def set_trajectory(trajectory, path):
    # path_dir = pathlib.Path(UPLOAD_FOLDER_ROOT) / session_id
    # path_dir.mkdir(parents=True, exist_ok=True)
    # path = path_dir / f"{button_id}-trajectory.pickle"
    with open(path, "wb") as f:
        dill.dump(trajectory, f)

    return path
