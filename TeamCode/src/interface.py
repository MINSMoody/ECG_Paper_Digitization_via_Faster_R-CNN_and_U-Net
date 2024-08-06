class AbstractModel:
    def train_model(self, data_folder, model_folder, verbose):
        """given folder of data, train a model and save it to a folder

        Args:
            data_folder (_type_): path to the datafolder
            model_folder (_type_): path to where the model should be saved
            verbose (bool): 

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError('This method must be implemented by a subclass.')
    
    @classmethod
    def from_folder(cls, model_folder, verbose):
        """Load a model from a folder as it was saved by train_model

        Args:
            model_folder (_type_): path to the folder where the model is saved
            verbose (_type_): _description_
        Returns:
            an instance of the class

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError('This method must be implemented by a subclass.')

class AbstractDigitizationModel(AbstractModel):
    def run_digitization_model(self, record, verbose):
        """Predict the signal timeseries of a record

        Args:
            record (str): path to record file
            verbose (_type_): _description_
        Returns:
            signal (np.array) : the predicted signal

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError('This method must be implemented by a subclass.')

    

class AbstractClassificationModel(AbstractModel):
    def run_classification_model(self, record, signal, verbose):
        """Predict the class of a record

        Args:
            record (str): path to record file
            signal (np.array): the signal predicted by our own digitization model
            verbose (_type_): _description_
        Returns:
            list[str] with the predicted classes

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError('This method must be implemented by a subclass.')

  