import json
import logging
from scipy.optimize import OptimizeResult


logger = logging.getLogger("CopPhil-EmpiricalBathymetry")


class Metadata:
    
    """ ** Work In Progress **
        Class to store metadata about the processing of the EmpiricalBathymetry class.
        Also, directs information to the logger.
        To use in any area of the code, simply import the metadata.Metadata class and create an instance with Metadata().
        You can use Metadata().add_info(key, value) for any additional information you wish to store.
        Or add your own methods to store specific information about the processing.
    """
    
    _instance = None

    # Singleton pattern - Only one instance of this class can exist. All Metadata imports will refer to the same instance.
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Metadata, cls).__new__(cls)
            cls._instance.data = {}
        return cls._instance


    def add_info(self, key: str, value: dict) -> None:
        """Add a dictionary of information to the metadata.

        Args:
            key (str): Key to store the dictionary under.
            value (dict): Dictionary of information to store.
        """
        self.data[key] = value


    def add_input_bands(self, obj) -> None:
        """Append metadata with input bands information.

        Args:
            obj (EmpiricalBathymetry): Instance of running EmpiricalBathymetry class.
        """
        
        band_map = [('BAND_I', obj._band_i),
                    ('BAND_J', obj._band_j)]

        self.data['INPUT_BANDS'] = {}

        for band_name, array_name in band_map:
            arr = obj._data_array.sel(band=array_name)
            self.data['INPUT_BANDS'][band_name] = {'NAME'  : array_name,
                                                   'SHAPE' : arr.shape,
                                                   'CRS'   : arr.rio.crs.to_string(),
                                                   'MIN'   : float(arr.min()),
                                                   'MAX'   : float(arr.max()),
                                                   'MEAN'  : float(arr.mean()),
                                                   'STD'   : float(arr.std())}
            # Send to logger
            logger.info(f'{band_name} SUMMARY : ')
            self.log_dict(self.data['INPUT_BANDS'][band_name])


    def log_dict(self, in_dict: dict) -> None:
        """Takes a dictionary, and prints out key-value pairs to the logger.

        Args:
            in_dict (dict): Dictionary of information to be logged.
        """
        for k,v in in_dict.items():
            logger.info(f'      {k} : {v}')
    

    def add_general(self, obj) -> None:
        """Append metadata with general information about processing and settings.

        Args:
            obj (EmpiricalBathymetry): Instance of running EmpiricalBathymetry class.
        """
        
        self.data['GENERAL'] = {'PROCESS_NAME'    : 'CopPhil-EmpiricalBathymetry',
                                'PROCESS_START'   : obj._start.strftime('%Y-%m-%d %H:%M:%S'),
                                'PROCESS_END'     : None,
                                'PROCESS_ELAPSED' : None,
                                'PROCESS_LABEL'   : obj._proc_label,
                                'N_FACTOR'        : obj._n,
                                'BAND_I_NAME'     : obj._band_i,
                                'BAND_J_NAME'     : obj._band_j,
                                'SIGMA_FILTER'    : obj._sigma_filter,
                                'MINMAX_FILTER'   : obj._minmax_filter}
        # Send to logger
        logger.info(f'CONFIGURATION SUMMARY : ')
        self.log_dict(self.data['GENERAL'])


    def add_raw_insitu(self, obj) -> None:
        """Append metadata with the raw, unaltered insitu data information.

        Args:
            obj (EmpiricalBathymetry): Instance of running EmpiricalBathymetry class.
        """
        
        self.data['RAW_INSITU'] = {'Z_MEAN' : obj._insitu_gdf['Z'].mean(),
                                   'Z_STD'  : obj._insitu_gdf['Z'].std(),
                                   'Z_MIN'  : obj._insitu_gdf['Z'].min(),
                                   'Z_MAX'  : obj._insitu_gdf['Z'].max(),
                                   'POINTS' : len(obj._insitu_gdf),
                                   'CRS'    : obj._insitu_gdf.crs.to_string()}
        logger.info(f'RAW INSITU SUMMARY : ')
        self.log_dict(self.data['RAW_INSITU'])


    def add_preprocessed_insitu(self, obj) -> None:
        """Append metadata with the pre-processed insitu data information.

        Args:
            obj (EmpiricalBathymetry): Instance of running EmpiricalBathymetry class.
        """
        
        self.data['PROCESSED_INSITU'] = {'Z_MEAN' : obj._insitu_gdf['Z'].mean(),
                                         'Z_STD' : obj._insitu_gdf['Z'].std(),
                                         'Z_MIN' : obj._insitu_gdf['Z'].min(),
                                         'Z_MAX' : obj._insitu_gdf['Z'].max(),
                                         'POINTS' : len(obj._insitu_gdf),
                                         'CRS' : obj._insitu_gdf.crs.to_string()}
        logger.info(f'PRE-PROCESSED INSITU SUMMARY : ')
        self.log_dict(self.data['PROCESSED_INSITU'])


    def add_optimise_calibration(self, result: OptimizeResult) -> None:
        """Append metadata with resuls of the calibration optimisation.

        Args:
            obj (OptimizeResult): Output of a scipy.optimize.minimize function.
        """
        
        self.data['CALIBRATION'] = {'SUCCESS?' : result['success'],
                                    'ITERATIONS' : result['nit'],
                                    'FINAL_RMSE' : result['fun'],
                                    'M0' : result['x'][0],
                                    'M1' : result['x'][1]}
        logger.info(f'CALIBRATION : ')
        self.log_dict(self.data['CALIBRATION'])


    def update_general_endtime(self, obj) -> None:
        
        """Update the values of the general metadata with the end time and elapsed time.

        Args:
            obj (EmpiricalBathymetry): Instance of running EmpiricalBathymetry class.
        """
        
        self.data['GENERAL']['PROCESS_END'] = obj._end.strftime('%Y-%m-%d %H:%M:%S')
        self.data['GENERAL']['PROCESS_ELAPSED'] = f'{(obj._end - obj._start)}s'
        logger.info('PROCESSING TIME : ' + self.data['GENERAL']['PROCESS_ELAPSED'])


    def export(self, file_path: str) -> None:
        
        """Write the metadata to a JSON file.

        Args:
            file_path (str): File path to write the metadata to.
        """
        
        with open(file_path, 'w') as f:
            json.dump(self.data, f, indent=4)