# -*- coding: utf-8 -*-
"""
Scenario testing: merging vehicle joining a platoon in the
customized 2-lane freeway simplified map sorely with carla
"""

# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import carla
import opencda.scenario_testing.utils.sim_api as sim_api
from opencda.core.common.cav_world import CavWorld
from opencda.scenario_testing.evaluations.evaluate_manager import EvaluationManager
from opencda.scenario_testing.utils.yaml_utils import load_yaml, save_yaml

def run_scenario(opt, config_yaml):
    try:
        scenario_params = load_yaml(config_yaml)

        # 创建仿真世界 
        cav_world = CavWorld(opt.apply_ml)

        # 创建场景管理器
        scenario_manager = sim_api.ScenarioManager(scenario_params,opt.apply_ml,opt.version,town='Town06',cav_world=cav_world)

        # 创建感知车
        single_cav_list = scenario_manager.create_vehicle_manager(opt,application=['single'],data_dump=False)

        # 创建交通流和背景车辆 
        traffic_manager, bg_veh_list = scenario_manager.create_traffic_carla()
        
        # 创建评估管理器 
        eval_manager = EvaluationManager(scenario_manager.cav_world,script_name='coop_town06',current_time=scenario_params['current_time'])
        
        scenario_manager.tick()     # 更新场景
        spectator = scenario_manager.world.get_spectator()

        # 初始化一个空的轨迹点列表
        rx = [[0],[0],[0]]
        ry = [[0],[0],[0]]
        while True:
            transform = single_cav_list[0].vehicle.get_transform()
            spectator.set_transform(carla.Transform(transform.location +carla.Location(z=100),carla.Rotation(pitch=-90)))   # 设置视角
            scenario_manager.tick()                         # 更新场景
            for i, single_cav in enumerate(single_cav_list):
                single_cav.update_info(opt,rx[i],ry[i])     # 根据仿真场景中的数据，更新自身的定位、感知、地图、V2X、控制器状态（位置速度）信息
                control,rx[i],ry[i] = single_cav.run_step() # 规划轨迹、得到下一时刻目标点、计算得到轨迹点
                single_cav.vehicle.apply_control(control)   # 根据控制指令实施控制
    finally:
        eval_manager.evaluate()
        if opt.record:
            scenario_manager.client.stop_recorder()
        scenario_manager.close()
        for v in single_cav_list:
            v.destroy()
        for bg_v in bg_veh_list:
            bg_v.destroy()
