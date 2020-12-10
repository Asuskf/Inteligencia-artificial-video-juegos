# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 13:11:21 2020

@author: david
"""

import json

class ParamsManager(object):
    
    def __init__(self, params_file):
        self.params = json.load(open(params_file, 'r'))
        
    def get_params(self):
        return self.params
    
    def get_agent_params(self):
        return self.params['agent']
    
    def get_environment_params(self):
        return self.params['environment']
    
    def update_agent_params(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.get_agent_params().keys():
                self.params['agent'][key] = value
    
    def export_agent_params(self, file_name):
        with open(file_name, 'w') as f:
            json.dump(self.params['agent'], f, indent=4, separators = (',', ':'), sort_keys=True)
            f.write("\n")
    
    def export_environment_params(self, file_name):
        with open(file_name, 'w') as f:
            json.dump(self.params['environment'], f, indent=4, separators = (',', ':'), sort_keys=True)
            f.write("\n")
        
if __name__ == "__main__":
    print("Probando fichero")
    param_file = "../parameters.json"
    manager = ParamsManager(param_file)
    agent_params = manager.get_agent_params()
    print("\n Los parametros del agente son:")
    for key, value in agent_params.items():
        print(key, ":", value)
    
    env_params = manager.get_environment_params()
    print("\n Los parametros del entorno son:")
    for key, value in env_params.items():
        print(key, ":", value)
        
    manager.update_agent_params(learning_rate = 0.01, gamma = 0.92)
    agent_params_updated = manager.get_agent_params()
    print("\n Los parametros del agente actulizados son:")
    for key, value in agent_params_updated.items():
        print(key, ":", value)
        
    print("Fin de prueba")