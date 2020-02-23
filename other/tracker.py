import os
from csv import writer

from agents import OptimalAgent


class Tracker():
    def __init__(self, agent: OptimalAgent):
        self.agent = agent

    @staticmethod
    def append_list_as_row(file_name, list_of_elem):
        # Open file in append mode
        with open(file_name, 'a+', newline='') as write_obj:
            # Create a writer object from csv module
            csv_writer = writer(write_obj)
            # Add contents of list as last row in the csv file
            csv_writer.writerow(list_of_elem)

    @staticmethod
    def create_file(file_name, cols):
        if not os.path.exists(file_name):
            to_write = ",".join(cols) + "\n"
            with open(file_name, "w") as f:
                f.write(to_write)

    def log_uniform(self):
        file_name = "results/uniform.csv"
        self.create_file(file_name, ["Ts", "Tt", "total_dist", "num_samples"])
        results = [self.agent.Ts, self.agent.Tt, self.agent.total_dist, self.agent.num_samples]
        self.append_list_as_row(file_name, results)