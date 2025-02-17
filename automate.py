#!/usr/bin/env ipython

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def validate_model(
    estimate_sensor1,
    estimate_sensor2,
    estimate_sensor3,
    estimate_sensor4,
    GT=[0, 0, 60],
    path="Z_up_1302_1601.csv",
):
    valid_sensor1, valid_sensor2, valid_sensor3, valid_sensor4, train_indexes = (
        singular_extract_valid(path)
    )
    simple_score1 = np.linalg.norm(valid_sensor1 - GT, axis=1).mean()
    print("No algo error sensor1: ", simple_score1)
    score_no_err_classical1 = np.linalg.norm(
        prediction(valid_sensor1, estimate_sensor1) - GT, axis=1
    ).mean()
    print("Classical Algo sensor1 Error is :", score_no_err_classical1)
    print("\n")

    simple_score2 = np.linalg.norm(valid_sensor2 - GT, axis=1).mean()
    print("No algo error sensor2: ", simple_score2)
    score_no_err_classical2 = np.linalg.norm(
        prediction(valid_sensor2, estimate_sensor2) - GT, axis=1
    ).mean()
    print("Classical Algo sensor2 Error is :", score_no_err_classical2)
    print("\n")

    simple_score3 = np.linalg.norm(valid_sensor3 - GT, axis=1).mean()
    print("No algo error sensor3: ", simple_score3)
    score_no_err_classical3 = np.linalg.norm(
        prediction(valid_sensor3, estimate_sensor3) - GT, axis=1
    ).mean()
    print("Classical Algo sensor3 Error is :", score_no_err_classical3)
    print("\n")

    simple_score4 = np.linalg.norm(valid_sensor4 - GT, axis=1).mean()
    print("No algo error sensor4: ", simple_score4)
    score_no_err_classical4 = np.linalg.norm(
        prediction(valid_sensor4, estimate_sensor4) - GT, axis=1
    ).mean()
    print("Classical Algo sensor4 Error is :", score_no_err_classical4)

    return (
        simple_score1,
        score_no_err_classical1,
        simple_score2,
        score_no_err_classical2,
        simple_score3,
        score_no_err_classical3,
        simple_score4,
        score_no_err_classical4,
    )


def prediction(gyro_input, M):
    b = M[:, 3]
    scale = np.identity(3) + M[0:3, 0:3]
    output = (gyro_input - b) @ np.linalg.inv(scale)
    return output


def singular_extract(path="Z_up_1302_1601.csv", axis="gyro_z", GT=[0, 0, 60]):
    """
    Extract the Gyro data and returns
    the gyro reading of the first 80 percept
    also remove the GT from the gyro
    """
    df = pd.read_csv(
        path,
        names=[
            "sensor",
            "acc_x",
            "acc_y",
            "acc_z",
            "gyro_x",
            "gyro_y",
            "gyro_z",
            "time",
        ],
    )
    df = df.dropna()
    print(df.head(1))
    percent = 80
    sensor1 = df[df["sensor"] == 1]
    train_indexes = int(len(sensor1) * (percent / 100))
    print(train_indexes)

    sensor1 = df[df["sensor"] == 1]
    sensor1 = sensor1.head(train_indexes)
    sensor1 = sensor1[["gyro_x", "gyro_y", "gyro_z"]] - GT
    sensor2 = df[df["sensor"] == 2]
    sensor2 = sensor2.head(train_indexes)
    sensor2 = sensor2[["gyro_x", "gyro_y", "gyro_z"]] - GT
    sensor3 = df[df["sensor"] == 3]
    sensor3 = sensor3.head(train_indexes)
    sensor3 = sensor3[["gyro_x", "gyro_y", "gyro_z"]] - GT
    sensor4 = df[df["sensor"] == 4]
    sensor4 = sensor4.head(train_indexes)
    sensor4 = sensor4[["gyro_x", "gyro_y", "gyro_z"]] - GT
    # barak = df[["gyro_x", "gyro_y", "gyro_z"]] - [0, 0, 60]
    return sensor1, sensor2, sensor3, sensor4, train_indexes


def singular_extract_valid(path="Z_up_1302_1601.csv"):
    """
    Extracts the data of the Gyros of a given file and
    returns the last 20 percent for validation
    """
    df = pd.read_csv(
        path,
        names=[
            "sensor",
            "acc_x",
            "acc_y",
            "acc_z",
            "gyro_x",
            "gyro_y",
            "gyro_z",
            "time",
        ],
    )
    df = df.dropna()
    print(df.head(1))
    percent = 20
    sensor1 = df[df["sensor"] == 1]
    train_indexes = int(len(sensor1) * (percent / 100))
    print(train_indexes)

    sensor1 = df[df["sensor"] == 1]
    sensor1 = sensor1.tail(train_indexes)
    sensor1 = sensor1[["gyro_x", "gyro_y", "gyro_z"]]
    sensor2 = df[df["sensor"] == 2]
    sensor2 = sensor2.tail(train_indexes)
    sensor2 = sensor2[["gyro_x", "gyro_y", "gyro_z"]]
    sensor3 = df[df["sensor"] == 3]
    sensor3 = sensor3.tail(train_indexes)
    sensor3 = sensor3[["gyro_x", "gyro_y", "gyro_z"]]
    sensor4 = df[df["sensor"] == 4]
    sensor4 = sensor4.tail(train_indexes)
    sensor4 = sensor4[["gyro_x", "gyro_y", "gyro_z"]]
    # barak = df[["gyro_x", "gyro_y", "gyro_z"]] - [0, 0, 60]
    return sensor1, sensor2, sensor3, sensor4, train_indexes


# One of the Sensors !
def get_estimate(
    sensor_x_minus,
    sensor_x_plus,
    sensor_y_minus,
    sensor_y_plus,
    sensor_z_minus,
    sensor_z_plus,
):
    """Input data from one of the sensors !"""
    A = np.array(
        [
            [-60, 60, 0, 0, 0, 0],
            [0, 0, -60, 60, 0, 0],
            [0, 0, 0, 0, -60, 60],
            [1, 1, 1, 1, 1, 1],
        ]
    )
    # np.vstack([barak,barak,barak,barak])
    z_sensor = np.hstack(
        [
            sensor_x_minus,
            sensor_x_plus,
            sensor_y_minus,
            sensor_y_plus,
            sensor_z_minus,
            sensor_z_plus,
        ]
    )  # .reshape(3, 6)
    z_sensor = z_sensor.reshape(-1, 3, 6, order="F")
    estimated_m = 1 / data_length1 * z_sensor @ A.T @ np.linalg.inv(A @ A.T)
    estimated_m = np.sum(estimated_m, axis=0)
    return estimated_m


# ysiw(
# :iedit regex !
sensor1_x_minus, sensor2_x_minus, sensor3_x_minus, sensor4_x_minus, data_length1 = (
    singular_extract(path="X_dn_1302_1601.csv", axis="gyro_x", GT=[-60, 0, 0])
)
sensor1_x_plus, sensor2_x_plus, sensor3_x_plus, sensor4_x_plus, data_length2 = (
    singular_extract(path="X_up_1302_1601.csv", axis="gyro_x", GT=[60, 0, 0])
)
sensor1_y_minus, sensor2_y_minus, sensor3_y_minus, sensor4_y_minus, data_length3 = (
    singular_extract(path="Y_dn_1302_1601.csv", axis="gyro_y", GT=[0, -60, 0])
)
sensor1_y_plus, sensor2_y_plus, sensor3_y_plus, sensor4_y_plus, data_length4 = (
    singular_extract(path="Y_up_1302_1601.csv", axis="gyro_y", GT=[0, 60, 0])
)
sensor1_z_minus, sensor2_z_minus, sensor3_z_minus, sensor4_z_minus, data_length5 = (
    singular_extract(path="Z_dn_1302_1601.csv", axis="gyro_z", GT=[0, 0, -60])
)
sensor1_z_plus, sensor2_z_plus, sensor3_z_plus, sensor4_z_plus, data_length6 = (
    singular_extract(path="Z_up_1302_1601.csv", axis="gyro_z", GT=[0, 0, 60])
)
#  sensor1_z @ (3*np.identity(3))
estimate_sensor1 = get_estimate(
    sensor1_x_minus,
    sensor1_x_plus,
    sensor1_y_minus,
    sensor1_y_plus,
    sensor1_z_minus,
    sensor1_z_plus,
)
estimate_sensor2 = get_estimate(
    sensor2_x_minus,
    sensor2_x_plus,
    sensor2_y_minus,
    sensor2_y_plus,
    sensor2_z_minus,
    sensor2_z_plus,
)
estimate_sensor3 = get_estimate(
    sensor3_x_minus,
    sensor3_x_plus,
    sensor3_y_minus,
    sensor3_y_plus,
    sensor3_z_minus,
    sensor3_z_plus,
)
estimate_sensor4 = get_estimate(
    sensor4_x_minus,
    sensor4_x_plus,
    sensor4_y_minus,
    sensor4_y_plus,
    sensor4_z_minus,
    sensor4_z_plus,
)

print("error estimation sensor 1 to 4 z plus direction ")
(
    simple_score1,
    score_no_err_classical1,
    simple_score2,
    score_no_err_classical2,
    simple_score3,
    score_no_err_classical3,
    simple_score4,
    score_no_err_classical4,
) = validate_model(
    estimate_sensor1,
    estimate_sensor2,
    estimate_sensor3,
    estimate_sensor4,
    GT=[0, 0, 60],
    path="Z_up_1302_1601.csv",
)
# 3
print("error estimation sensor 1 to 4 z minus direction")
(
    simple_score1,
    score_no_err_classical1,
    simple_score2,
    score_no_err_classical2,
    simple_score3,
    score_no_err_classical3,
    simple_score4,
    score_no_err_classical4,
) = validate_model(
    estimate_sensor1,
    estimate_sensor2,
    estimate_sensor3,
    estimate_sensor4,
    GT=[0, 0, -60],
    path="Z_dn_1302_1601.csv",
)

species = ("Sensor 1", "Sensor 2", "Sensor 3", "Sensor 4")
penguin_means = {
    "No Algo Err": (simple_score1, simple_score2, simple_score3, simple_score4),
    "Classical Algo Err": (
        score_no_err_classical1,
        score_no_err_classical2,
        score_no_err_classical3,
        score_no_err_classical4,
    ),
    "ML Algo": (0.4, 0.5, 0.1, 0.5),
}

x = np.arange(len(species))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout="constrained")

for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel("Error in norm2 (deg/sec)")
ax.set_title("Performance Comparsion No Algo , classical , ML, Z Plus")
ax.set_xticks(x + width, species)
ax.legend(loc="upper left", ncols=3)
ax.set_ylim(0, 4)

plt.show()
