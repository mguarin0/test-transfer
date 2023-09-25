import cPickle
import os

with open(os.path.join("train.cpkl"), "r") as file:
  train = cPickle.load(file) 

"""
>>> type(train)
<type 'tuple'>

>>> len(train)
2

>>> type(train[0])
<type 'list'>

>>> type(train[1])
<type 'list'>

>>> len(train[0])
175

>>> len(train[1])
175

>>> train[0]
['2B42', '4CPA', '3CPH', '1EZU', '1R8S', '1DE4', '1XU1', '1FAK', '1AKJ', '1IBR', '1BJ1', '2HMI', '1WEJ', '1XD3', '1F51', '1E6J', '7CEI', '1ZHH', '1MLC', '2SNI', '1GHQ', '1HE1', '1YVB', '2A5T', '1T6B', '2A9K', '1Z0K', '1ZHI', '1LFD', '1AY7', '1I9R', '1GRN', '1S1Q', '2J7P', '1US7', '2PCC', '1B6C', '1AZS', '2IDO', '1EER', '1BGX', '1GPW', '1SBB', '1R0R', '1Z5Y', '2FD6', '1GP2', '1JTG', '1PXV', '2C0L', '2CFH', '1JPS', '1HCF', '1XQS', '1CLV', '2O3B', '3SGQ', '1ATN', '2HLE', '1F6M', '1M10', '1OPH', '3D5S', '1QFW', '1R6Q', '1Y64', 'BOYV', '2UUY', '1KXP', '1WDW', '1ML0', '9QFW', '1BUH', '1FQ1', '1PVH', '1GL1', '3BP8', '1KLU', '1TMQ', '1FCC', '1NSN', '1JIW', '1E6E', '1DFJ', '2I9B', '2AYO', '1ACB', '1K74', '1OFU', '2HQS', '2Z0E', '1RV6', '1IQD', '2OUL', '2O8V', '1EWY', '2B4J', '1FFW', '1IB1', '1GLA', '2OOR', '1HIA', '1E96', '2OT3', '2VDB', '1RLB', '1HE8', '1QA9', '2H7V', '1I2M', '1N2C', '1OC0', '1ZLI', '1KXQ', '1KAC', '2I25', '1KKL', '1WQ1', '1SYX', '1EAW', '1JZD', '2J0T', '1FSK', '1K5D', '1AK4', '1F34', '1BVN', '1JK9', '1MAH', '1FQJ', '1VFB', '1A2K', '1J2J', '1NCA', '1GXD', '1UDI', '1D6R', '2HRK', '1H9D', '1NW9', '2OOB', '2SIC', '1FLE', '1IRA', '2MTA', '1BKD', '1OYV', '1AVX', '2FJU', '1MQ8', '2BTF', '2AJF', '1K4C', '1E4K', '2JEL', '1CGI', '1ZM4', '1JWH', '2G77', '1I4D', '1KTZ', '1AHW', '1GCQ', '1BVK', '1DQJ', '2OZA', '2ABZ', '2VIS', '1EFN', '1FC2', '1JMO', '1H1V', '1IJK', '2NZ8', '1PPE']

>>> type(train[1][0])
<type 'dict'>

>>> train[1][0].keys()
['r_vertex', 'l_vertex', 'complex_code', 'l_edge', 'label', 'r_hood_indices', 'r_edge', 'l_hood_indices']

>>> train[1][0]['r_vertex'].shape
(362, 70)

>>> train[1][0]['l_vertex'].shape
(185, 70)

>>> train[1][0]["complex_code"]
'2B42'

>>> train[1][0]["r_edge"].shape
(362, 20, 2)

>>> train[1][0]["l_edge"].shape
(185, 20, 2)

>>> train[1][0]["label"].shape
(1683, 3)

>>> train[1][0]["r_hood_indices"].shape
(362, 20, 1)

>>> train[1][0]["l_hood_indices"].shape
(185, 20, 1)

# run all same stuff with another index
>>> train[1][10]["r_vertex"].shape
(431, 70)

>>> train[1][10]["l_vertex"].shape
(189, 70)

>>> train[1][10]["r_edge"].shape
(431, 20, 2)
"""
  

