import multiprocessing
import school_model
import itertools

class BatchrunnerMP():
    def __init__(self, model, map_path, schedule_path, max_steps, nr_processes = None, iterations=1, fixed_parameters=None, variable_parameters=None, param_df=None):
        if (fixed_parameters is None) and (variable_parameters is None) and (param_df is None):
            raise ValueError('All parameter inputs cannot be None!')
        self.model = model
        self.max_steps = max_steps
        self.iterations = iterations
        self.fixed_parameters = fixed_parameters
        self.variable_parameters = variable_parameters
        self.param_df = param_df
        self.map_path = map_path
        self.schedule_path = schedule_path
        
        
        
        if nr_processes == None:
            #identifies the number of processors available on users machine
            available_processors = multiprocessing.cpu_count()
            self.processes = available_processors
            print ("Your system has {} available processors.".format(self.processes))
        else:
            self.processes = nr_processes
            
        

    
    
    def run_all(self):
        if (self.param_df is None):
            run_iter_args = self._make_model_args()
        else:
            run_iter_args = self._make_args_from_df()
            
        

            
        
        
        
        pool = multiprocessing.Pool(processes = self.processes)

        results = {}
        if self.processes > 1:
            for params, model_data in pool.imap_unordered(self.run_wrapper, run_iter_args):
                results[str(params)] = model_data
            
        else:
            for run in run_iter_args:
                params, model_data = self.run_wrapper(run)
                results[str(params)] = model_data
        
        
        return results
        
        
        
    @staticmethod       
    def run_wrapper(iter_args):
        model_i = iter_args[0]
        kwargs = iter_args[1]
        max_steps = iter_args[2]
        iteration = iter_args[3]
        cur_model = model_i(**kwargs)     
        while cur_model.running and cur_model.schedule.steps < max_steps:
            cur_model.step()
        kwargs["iteration"] = iteration
        return kwargs, (cur_model.datacollector.get_model_vars_dataframe(), cur_model.datacollector.get_agent_vars_dataframe())   

    
             


    def _make_model_args(self):
        '''
        Generate all combination of model arguments from dictionary of fixed parameters and variable parameters
        Args:
            fixed_parameter: a dictionary of fixed parameters
                an example would be {'param_name1': value1, 'param_name2' value2}
            variable_parameter: a dictionary of variable parameters
                an example would be {'param_name1': [1, 2, 3], 'param_name2': ['value1', 'value2']]
            max_steps: maximum step the model can run before force termination
            iterations: number of iterations of model run
        Returns:
            a list of model parameters that can be applied to model with run_wrapper
        '''
        def unnest(lst):
            out = []
            for i in lst:
                try:
                    out += list(i)
                except TypeError:
                    out += [i]
            return out

        var_params = list(self.variable_parameters.values())

        # handle case when variable parameters is empty
        try:
            cur_param = var_params[0]
        except:
            cur_param = [[]]
            
        
        for p in var_params[1:]:

            cur_param = list(itertools.product(cur_param, p))
            cur_param = list(map(unnest, cur_param))

        var_param_names = list(self.variable_parameters.keys())

        run_iter_args = []
        
        # handle case when variable parameters has only one variable arg
        if not isinstance(cur_param[0], list):
            cur_param = list(map(lambda x: [x], cur_param))
            
        # loop through parameter combinations and build input args for model
        for ite in range(self.iterations):
            for params in cur_param:
                cur_iter_args = []
                cur_args = {}
                cur_args['map_path'] = self.map_path
                cur_args['schedule_path'] = self.schedule_path
                for i in range(len(params)):
                    cur_args[var_param_names[i]] = params[i]
                cur_args =  {**self.fixed_parameters, **cur_args}
                cur_iter_args.append(self.model)
                cur_iter_args.append(cur_args)
                cur_iter_args.append(self.max_steps)
                cur_iter_args.append(ite)
                run_iter_args.append(cur_iter_args)
        return run_iter_args





    def _make_args_from_df(self):
        '''
        Generate appropriate model arguments from dataframe
        Args:
            param_df: dataframe containing model run params in each row, normally pulled from ModelParameter google sheet
            max_steps: maximum step the model can run before force termination
            iterations: number of iterations of model run
        Returns:
            a list of model parameters that can be applied to model with run_wrapper
        '''
        def get_args(run_iter_args, row, cur_iter):
            cur_iter_args = []
            cur_iter_args.append(self.model)
            
            cur_args = row.to_dict()
            cur_args['map_path'] = self.map_path
            cur_args['schedule_path'] = self.schedule_path
            
            
            cur_iter_args.append(cur_args)
            cur_iter_args.append(self.max_steps)
            cur_iter_args.append(cur_iter)
            
            
            run_iter_args.append(cur_iter_args)



        run_iter_args = []
        for cur_iter in range(self.iterations):
            self.param_df.apply(lambda row: get_args(run_iter_args, row, cur_iter), axis=1)
        return run_iter_args
    
    
    


