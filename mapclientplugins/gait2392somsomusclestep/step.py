"""
MAP Client Plugin Step
"""
import json

from PySide6 import QtGui

from mapclient.mountpoints.workflowstep import WorkflowStepMountPoint
from mapclientplugins.gait2392somsomusclestep.configuredialog import \
    ConfigureDialog
from mapclientplugins.gait2392somsomusclestep.gait2392musclecustsomso import \
    Gait2392MuscleCustomiser


class FieldworkGait2392SomsoMuscleStep(WorkflowStepMountPoint):
    """
    MAP Client plugin for customising the OpenSim Gait2392 model muscle points

    Inputs
    ------
    gias-lowerlimb : GIAS3 LowerlimbAtlas instance
        Lower limb model with customised lower limb bone geometry and pose
    osimmodel : OpenSim model instance
        The opensim model to modify. Should be output from a step that
        modified the body geometries.

    Outputs
    -------
    osimmodel : OpenSim model instance
        Modified opensim model
    """

    def __init__(self, location):
        super(FieldworkGait2392SomsoMuscleStep, self).__init__(
            'Gait2392 SOMSO Muscle', location)
        # A step cannot be executed until it has been configured.
        self._configured = False
        self._category = 'OpenSim'
        # Add any other initialisation code here:
        self._icon = QtGui.QImage(
            ':/fieldworkgait2392musclehmfstep/images/morphometric.png')
        # Ports:
        self.addPort(
            ('http://physiomeproject.org/workflow/1.0/rdf-schema#port',
             'http://physiomeproject.org/workflow/1.0/rdf-schema#uses',
             'http://physiomeproject.org/workflow/1.0/rdf-schema#gias-'
             'lowerlimb'))
        self.addPort(
            ('http://physiomeproject.org/workflow/1.0/rdf-schema#port',
             'http://physiomeproject.org/workflow/1.0/rdf-schema#uses',
             'http://physiomeproject.org/workflow/1.0/rdf-schema#osimmodel'))
        self.addPort(
            ('http://physiomeproject.org/workflow/1.0/rdf-schema#port',
             'http://physiomeproject.org/workflow/1.0/rdf-schema#uses',
             'http://physiomeproject.org/workflow/1.0/rdf-schema#landmarks'))
        # 'ju#fieldworkmodeldict'))
        self.addPort(
            ('http://physiomeproject.org/workflow/1.0/rdf-schema#port',
             'http://physiomeproject.org/workflow/1.0/rdf-schema#provides',
             'http://physiomeproject.org/workflow/1.0/rdf-schema#osimmodel'))

        # Config:
        self._config = {
            'identifier': '', 'osim_output_dir': './', 'in_unit': 'mm',
            'out_unit': 'm', 'write_osim_file': True, 'update_knee_splines': False, 'static_vas': False,
            'update_max_iso_forces': True, 'subject_height': '', 'subject_mass': ''
        }

        self._g2392_somso_muscle = Gait2392MuscleCustomiser(self._config)
        self._g2392_somso_muscle.set_workflow_location(self._location)

    def execute(self):
        """
        Add your code here that will kick off the execution of the step.
        Make sure you call the _doneExecution() method when finished. This
        method may be connected up to a button in a widget for example.
        """
        self._g2392_somso_muscle.config = self._config
        self._g2392_somso_muscle.customise()
        self._doneExecution()

    def setPortData(self, index, dataIn):
        """
        Add your code here that will set the appropriate objects for this step.
        The index is the index of the port in the port list.  If there is only
        one uses port for this step then the index can be ignored.
        """
        if index == 0:
            # http://physiomeproject.org/workflow/1.0/rdf-schema#gias-lowerlimb
            self._g2392_somso_muscle.ll = dataIn
        elif index == 1:
            # http://physiomeproject.org/workflow/1.0/rdf-schema#osimmodel
            self._g2392_somso_muscle.set_osim_model(dataIn)
        elif index == 2:
            self._g2392_somso_muscle.landmarks = dataIn

    def getPortData(self, index):
        """
        Add your code here that will return the appropriate objects for this
        step. The index is the index of the port in the port list.  If there is
        only one provides port for this step then the index can be ignored.
        """
        # http://physiomeproject.org/workflow/1.0/rdf-schema#osimmodel
        return self._g2392_somso_muscle.gias_osimmodel._model

    def configure(self):
        """
        This function will be called when the configure icon on the step is
        clicked.  It is appropriate to display a configuration dialog at this
        time.  If the conditions for the configuration of this step are complete
        then set:
            self._configured = True
        """
        dlg = ConfigureDialog()
        dlg.set_workflow_location(self._location)
        dlg.identifierOccursCount = self._identifierOccursCount
        dlg.setConfig(self._config)
        dlg.validate()
        dlg.setModal(True)

        if dlg.exec_():
            self._config = dlg.getConfig()

        self._configured = dlg.validate()
        self._configuredObserver()

    def getIdentifier(self):
        """
        The identifier is a string that must be unique within a workflow.
        """
        return self._config['identifier']

    def setIdentifier(self, identifier):
        """
        The framework will set the identifier for this step when it is loaded.
        """
        self._config['identifier'] = identifier

    def serialize(self):
        """
        Add code to serialize this step to string.  This method should
        implement the opposite of 'deserialize'.
        """
        return json.dumps(self._config, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)

    def deserialize(self, string):
        """
        Add code to deserialize this step from string.  This method should
        implement the opposite of 'serialize'.
        """
        self._config.update(json.loads(string))

        d = ConfigureDialog()
        d.set_workflow_location(self._location)
        d.identifierOccursCount = self._identifierOccursCount
        d.setConfig(self._config)
        self._configured = d.validate()
