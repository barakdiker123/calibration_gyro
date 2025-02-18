#!/usr/bin/env ipython
import pandas as pd
from automate import singular_extract


def create_data():
    sensor1_x_minus, sensor2_x_minus, sensor3_x_minus, sensor4_x_minus, data_length1 = (
        singular_extract(path="X_dn_1302_1601.csv",
                         GT=[0, 0, 0])
    )
    gt1 = pd.DataFrame([[-60, 0, 0] for i in range(data_length1)],
                       columns=['gt_x', 'gt_y', 'gt_z'])
    sensor1_x_plus, sensor2_x_plus, sensor3_x_plus, sensor4_x_plus, data_length2 = (
        singular_extract(path="X_up_1302_1601.csv",
                         GT=[0, 0, 0])
    )
    gt2 = pd.DataFrame([[60, 0, 0] for i in range(data_length2)],
                       columns=['gt_x', 'gt_y', 'gt_z'])
    sensor1_y_minus, sensor2_y_minus, sensor3_y_minus, sensor4_y_minus, data_length3 = (
        singular_extract(path="Y_dn_1302_1601.csv",
                         GT=[0, 0, 0])
    )
    gt3 = pd.DataFrame([[0, -60, 0] for i in range(data_length3)],
                       columns=['gt_x', 'gt_y', 'gt_z'])
    sensor1_y_plus, sensor2_y_plus, sensor3_y_plus, sensor4_y_plus, data_length4 = (
        singular_extract(path="Y_up_1302_1601.csv",
                         GT=[0, 0, 0])
    )
    gt4 = pd.DataFrame([[0, 60, 0] for i in range(data_length4)],
                       columns=['gt_x', 'gt_y', 'gt_z'])
    sensor1_z_minus, sensor2_z_minus, sensor3_z_minus, sensor4_z_minus, data_length5 = (
        singular_extract(path="Z_dn_1302_1601.csv",
                         GT=[0, 0, 0])
    )
    gt5 = pd.DataFrame([[0, 0, -60] for i in range(data_length5)],
                       columns=['gt_x', 'gt_y', 'gt_z'])
    sensor1_z_plus, sensor2_z_plus, sensor3_z_plus, sensor4_z_plus, data_length6 = (
        singular_extract(path="Z_up_1302_1601.csv",
                         GT=[0, 0, 0])
    )
    gt6 = pd.DataFrame([[0, 0, 60] for i in range(data_length6)],
                       columns=['gt_x', 'gt_y', 'gt_z'])
    sensor1_data = pd.concat([sensor1_x_minus, sensor1_x_plus,
                              sensor1_y_minus, sensor1_y_plus,
                              sensor1_z_minus, sensor1_z_plus])
    sensor2_data = pd.concat([sensor2_x_minus, sensor2_x_plus,
                              sensor2_y_minus, sensor2_y_plus,
                              sensor2_z_minus, sensor2_z_plus])
    sensor3_data = pd.concat([sensor3_x_minus, sensor3_x_plus,
                              sensor3_y_minus, sensor3_y_plus,
                              sensor3_z_minus, sensor3_z_plus])
    sensor4_data = pd.concat([sensor4_x_minus, sensor4_x_plus,
                              sensor4_y_minus, sensor4_y_plus,
                              sensor4_z_minus, sensor4_z_plus])
    gt = pd.concat([gt1, gt2, gt3, gt4, gt5, gt6])
    return sensor1_data, sensor2_data, sensor3_data, sensor4_data, gt


sensor1_data, sensor2_data, sensor3_data, sensor4_data, gt = create_data()
