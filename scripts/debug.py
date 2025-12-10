import opengate as gate
from opengate.contrib.pet.philipsvereos import *

sim = gate.Simulation()

create_material(sim)
add_pet(sim, debug=True)
add_table(sim)

sim.visu = True
sim.run()