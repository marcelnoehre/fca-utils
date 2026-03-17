from data.parser import Parser
from linear_extension.sat_realizer import SatRealizer
from linear_extension.additive_realizer import AdditiveRealizer

parser = Parser()
context = parser.decode_cxt('./data/standard/B3.cxt')
print('##### Formal Context #####')
print(context.print_data())

print('##### SAT-Realizer #####')
sat_realizer = SatRealizer(context)
dim, sat_R = sat_realizer.realizer()
for le in sat_R:
    print(le)

print('##### Additive-Realizer #####')
add_realizer = AdditiveRealizer(context, dim)
dim, add_R = add_realizer.realizer()
for le in add_R:
    print(le)