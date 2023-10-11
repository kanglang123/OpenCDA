# -*- coding: utf-8 -*-
"""
Scenario testing: merging vehicle joining a platoon in the
customized 2-lane freeway simplified map sorely with carla
"""
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib
import os

import carla

import opencda.scenario_testing.utils.sim_api as sim_api
from opencda.core.common.cav_world import CavWorld
from opencda.scenario_testing.evaluations.evaluate_manager import \
    EvaluationManager
from opencda.scenario_testing.utils.yaml_utils import load_yaml, save_yaml


def run_scenario(opt, config_yaml):
    try:
        scenario_params = load_yaml(config_yaml)

        # create CAV world
        cav_world = CavWorld(opt.apply_ml)

        # create scenario manager
        scenario_manager = sim_api.ScenarioManager(scenario_params,
                                                   opt.apply_ml,
                                                   opt.version,
                                                   town='Town06',
                                                   cav_world=cav_world)

        if opt.record:
            scenario_manager.client. \
                start_recorder("single_town06_carla.log", True)
        # 创建感知车
        single_cav_list = scenario_manager.create_vehicle_manager(opt,application=['single'],data_dump=False)
        scenario_manager.tick()     # 更新场景
        
        # 创建RSU
        # rsu_list = scenario_manager.create_rsu_manager(data_dump=True)

        # 创建交通流和背景车辆 create background traffic in carla 
        traffic_manager, bg_veh_list = scenario_manager.create_traffic_carla()
        scenario_manager.tick()     # 更新场景

        # 创建评估管理器 create evaluation manager 
        eval_manager = EvaluationManager(scenario_manager.cav_world,
                              script_name='coop_town06',
                              current_time=scenario_params['current_time'])

        spectator = scenario_manager.world.get_spectator()

        # 创建数据收集协议 save the data collection protocol to the folder 
        # current_path = os.path.dirname(os.path.realpath(__file__))
        # save_yaml_name = os.path.join(current_path,
        #                               '../../data_dumping',
        #                               scenario_params['current_time'],
        #                               'data_protocol.yaml')
        # save_yaml(scenario_params, save_yaml_name)
        rx = [0]
        ry = [0]
        while True:
            scenario_manager.tick()     # 更新场景
            transform = single_cav_list[0].vehicle.get_transform()
            spectator.set_transform(carla.Transform(transform.location +carla.Location(z=100),carla.Rotation(pitch=-90)))

            scenario_manager.tick()     # 更新场景
            
            for i, single_cav in enumerate(single_cav_list):
                single_cav.update_info(opt,rx,ry)
                control,rx,ry = single_cav.run_step()
                single_cav.vehicle.apply_control(control)

            # for rsu in rsu_list:
            #     rsu.update_info()
            #     rsu.run_step()

    finally:
        eval_manager.evaluate()

        if opt.record:
            scenario_manager.client.stop_recorder()

        scenario_manager.close()

        for v in single_cav_list:
            v.destroy()
        # for r in rsu_list:
        #     r.destroy()
        for bg_v in bg_veh_list:
            bg_v.destroy()
