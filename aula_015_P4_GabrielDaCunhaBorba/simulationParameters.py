from typing import Any, Dict, List, Sequence, Union
import json
import copy

class SimulationParameters():
    def __init__(self) -> None:
    #     self.step_size = 0.0
    #     self.number_steps = 0

    #     self.particle_mass = 0.0
    #     self.particle_hardness = 0.0
    #     # self.particle_radius = 0.0
    #     self.particle_half_xSize = 0.0
    #     self.particle_half_ySize = 0.0
    #     self.particle_coords = []
    #     self.particle_external_force = []
    #     self.particle_restricted = []
    #     self.particle_connection: Union[List,Dict] = []
        self.x_dist = 0.0
        self.y_dist = 0.0
        self.ij_pos = []
        self.connect = []
        self.boundary_condition = []

    # def __setattr__(self, name: str, value: Any) -> None:
    #     super().__setattr__(name, value)
    #     if name == 'particle_connection' and isinstance(value, dict):
    #         aux = copy.deepcopy(value)
    #         for i in range(1,len(self.particle_connection)+1):
    #             try:
    #                 aux[i] += [0]*(4-len(aux[i]))
    #             except KeyError:
    #                 pass
    #         self.particle_connection = [[len(v)-v.count(0),*v] for (k, v) in sorted(aux.items())]

    def __str__(self) -> str:
#         return f'\
# {self.step_size=}\n\
# {self.number_steps=}\n\
# {self.particle_mass=}\n\
# {self.particle_hardness=}\n\
# {self.particle_half_xSize}\n\
# {self.particle_half_ySize}\n\
# {self.particle_coords=}\n\
# {self.particle_external_force=}\n\
# {self.particle_restricted=}\n\
# {self.particle_connection=}'
        return f'\
{self.x_dist}\n\
{self.y_dist}\n\
{self.connect}\n\
{self.boundary_condition}\n'

    def saveAsJson(self, archiveName):
        # NOTE - writing the point cloud on a file
        with open(archiveName, 'w') as writeFile:
            json.dump(self, writeFile, default= lambda o: o.__dict__,indent=4)

    def loadFromJson(self, archiveName):
        # NOTE - reading the point cloud from a file
        content = {}
        with open(archiveName, 'r') as readFile:
            content: Dict = json.load(readFile)
            for k, v in content.items():
                try:
                    print(k)
                    vType = type(self.__dict__.get(k,None))
                    parsedV = vType(v)
                    self.__dict__[k] = parsedV
                except KeyError:
                    pass

# if __name__ == "__main__":
#     a = SimulationParameters()
#     a.loadFromJson("intest.json")
#     a.saveAsJson("outtest.json")
