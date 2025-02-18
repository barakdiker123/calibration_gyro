#!/usr/bin/env ipython

from automate import *
import torch


def simple_run(test_path="Z_dn_1302_1601.csv", GT_test=[0, 0, -60]):
    # ysiw(
    # :iedit regex !
    sensor1_x_minus, sensor2_x_minus, sensor3_x_minus, sensor4_x_minus, data_length1 = (
        singular_extract(path="X_dn_1302_1601.csv",
                         GT=[-60, 0, 0])
    )
    sensor1_x_plus, sensor2_x_plus, sensor3_x_plus, sensor4_x_plus, data_length2 = (
        singular_extract(path="X_up_1302_1601.csv",
                         GT=[60, 0, 0])
    )
    sensor1_y_minus, sensor2_y_minus, sensor3_y_minus, sensor4_y_minus, data_length3 = (
        singular_extract(path="Y_dn_1302_1601.csv",
                         GT=[0, -60, 0])
    )
    sensor1_y_plus, sensor2_y_plus, sensor3_y_plus, sensor4_y_plus, data_length4 = (
        singular_extract(path="Y_up_1302_1601.csv",
                         GT=[0, 60, 0])
    )
    sensor1_z_minus, sensor2_z_minus, sensor3_z_minus, sensor4_z_minus, data_length5 = (
        singular_extract(path="Z_dn_1302_1601.csv",
                         GT=[0, 0, -60])
    )
    sensor1_z_plus, sensor2_z_plus, sensor3_z_plus, sensor4_z_plus, data_length6 = (
        singular_extract(path="Z_up_1302_1601.csv",
                         GT=[0, 0, 60])
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

   # print("error estimation sensor 1 to 4 z plus direction ")
   # (
   #    simple_score1,
   #    score_no_err_classical1,
   #    simple_score2,
   #    score_no_err_classical2,
   #    simple_score3,
   #    score_no_err_classical3,
   #    simple_score4,
   #    score_no_err_classical4,
   # ) = validate_model(
   #    estimate_sensor1,
   #    estimate_sensor2,
   #    estimate_sensor3,
   #    estimate_sensor4,
   #    GT=[0, 0, 60],
   #    path="Z_up_1302_1601.csv",
   # )
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
        # GT=[0, 0, -60],
        # path="Z_dn_1302_1601.csv",
        GT=GT_test,
        path=test_path,
    )

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu")
    model_sensor1 = VectorPredictor().to(device)
    model_sensor1.load_state_dict(torch.load(
        "model_sensor1.pth", weights_only=True))

    model_sensor2 = VectorPredictor().to(device)
    model_sensor2.load_state_dict(torch.load(
        "model_sensor2.pth", weights_only=True))

    model_sensor3 = VectorPredictor().to(device)
    model_sensor3.load_state_dict(torch.load(
        "model_sensor3.pth", weights_only=True))

    model_sensor4 = VectorPredictor().to(device)
    model_sensor4.load_state_dict(torch.load(
        "model_sensor4.pth", weights_only=True))

    (score_ml_sensor1, score_ml_sensor2,
     score_ml_sensor3, score_ml_sensor4) = validate_deep_learning_model(
        model_sensor1,
        model_sensor2,
        model_sensor3,
        model_sensor4,
        # GT=[0, 0, 60],
        # path="Z_up_1302_1601.csv")
        GT=GT_test,
        path=test_path)
    print(score_ml_sensor1)

    species = ("Sensor 1", "Sensor 2", "Sensor 3", "Sensor 4")
    penguin_means = {
        "No Algo Err": (simple_score1, simple_score2, simple_score3, simple_score4),
        "Classical Algo Err": (
            score_no_err_classical1,
            score_no_err_classical2,
            score_no_err_classical3,
            score_no_err_classical4,
        ),
        "ML Algo": (score_ml_sensor1, score_ml_sensor2,
                    score_ml_sensor3, score_ml_sensor4),
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

# GT=[0, 0, 60],
# path="Z_up_1302_1601.csv"
# path="X_dn_1302_1601.csv",
#   GT=[-60, 0, 0]


simple_run()
# simple_run("Z_up_1302_1601.csv", [0, 0, 60])
# simple_run("X_dn_1302_1601.csv", [-60, 0, 0])
# simple_run("X_up_1302_1601.csv", [60, 0, 0])
# simple_run("Y_dn_1302_1601.csv", [0, -60, 0])
# simple_run("Y_up_1302_1601.csv", [0, 60, 0])
