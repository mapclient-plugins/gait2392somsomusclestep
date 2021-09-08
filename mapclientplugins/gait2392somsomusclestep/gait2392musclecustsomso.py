"""
Module for customising opensim segmented muscle points 
"""
import os
import numpy as np
import copy
from gias2.musculoskeletal.bonemodels import bonemodels
from gias2.musculoskeletal import osim
from mapclientplugins.gait2392somsomusclestep.muscleVolumeCalculator import muscleVolumeCalculator
import re
import math
import json
from numpy import pi
from scipy.interpolate import interp1d

SELF_DIR = os.path.split(__file__)[0]
DATA_DIR = os.path.join(SELF_DIR, 'data/node_numbers/')
TEMPLATE_OSIM_PATH = os.path.join(SELF_DIR, 'data',
                                  'gait2392_simbody_wrap.osim')

VALID_SEGS = {'pelvis', 'femur-l', 'femur-r', 'tibia-l', 'tibia-r'}
OSIM_FILENAME = 'gait2392_simbody.osim'
VALID_UNITS = ('nm', 'um', 'mm', 'cm', 'm', 'km')

TIBFIB_SUBMESHES = ('tibia', 'fibula')
TIBFIB_SUBMESH_ELEMS = {'tibia': range(0, 46), 'fibula': range(46, 88), }
TIBFIB_BASISTYPES = {'tri10': 'simplex_L3_L3', 'quad44': 'quad_L3_L3'}


def dim_unit_scaling(in_unit, out_unit):
    """
    Calculate the scaling factor to convert from the input unit (in_unit) to
    the output unit (out_unit). in_unit and out_unit must be a string and one
    of ['nm', 'um', 'mm', 'cm', 'm', 'km']. 

    inputs
    ======
    in_unit : str
        Input unit
    out_unit :str
        Output unit

    returns
    =======
    scaling_factor : float
    """

    unit_vals = {
        'nm': 1e-9,
        'um': 1e-6,
        'mm': 1e-3,
        'cm': 1e-2,
        'm': 1.0,
        'km': 1e3,
    }

    if in_unit not in unit_vals:
        raise ValueError(
            'Invalid input unit {}. Must be one of {}'.format(
                in_unit, list(unit_vals.keys())
            )
        )
    if out_unit not in unit_vals:
        raise ValueError(
            'Invalid input unit {}. Must be one of {}'.format(
                in_unit, list(unit_vals.keys())
            )
        )

    return unit_vals[in_unit] / unit_vals[out_unit]


def update_femur_opensim_acs(femur_model):
    femur_model.acs.update(
        *bonemodels.model_alignment.createFemurACSOpenSim(
            femur_model.landmarks['femur-HC'],
            femur_model.landmarks['femur-MEC'],
            femur_model.landmarks['femur-LEC'],
            side=femur_model.side
        )
    )


def update_tibiafibula_opensim_acs(tibiafibula_model):
    tibiafibula_model.acs.update(
        *bonemodels.model_alignment.createTibiaFibulaACSOpenSim(
            tibiafibula_model.landmarks['tibiafibula-MM'],
            tibiafibula_model.landmarks['tibiafibula-LM'],
            tibiafibula_model.landmarks['tibiafibula-MC'],
            tibiafibula_model.landmarks['tibiafibula-LC'],
            side=tibiafibula_model.side
        )
    )


def split_tibia_fibula_gfs(tib_fib_gf):
    tib = tib_fib_gf.makeGFFromElements(
        'tibia',
        TIBFIB_SUBMESH_ELEMS['tibia'],
        TIBFIB_BASISTYPES,
    )
    fib = tib_fib_gf.makeGFFromElements(
        'fibula',
        TIBFIB_SUBMESH_ELEMS['fibula'],
        TIBFIB_BASISTYPES,
    )

    return tib, fib


def local_osim_2_global(body, model):
    # find the knee angle
    knee = model.joints['knee_l']
    kneeAngle = model.joints['knee_l'].coordSets['knee_angle_l'].defaultValue
    knee_lTrans = np.zeros(3)

    # get the spline values
    trans1X = knee.getSimmSplineParams('translation1')[0]
    trans1Y = knee.getSimmSplineParams('translation1')[1]
    f = interp1d(trans1X, trans1Y, kind='cubic')

    knee_lTrans[0] = f(kneeAngle)

    trans2X = knee.getSimmSplineParams('translation2')[0]
    trans2Y = knee.getSimmSplineParams('translation2')[1]
    f2 = interp1d(trans2X, trans2Y, kind='cubic')
    knee_lTrans[1] = f2(kneeAngle)

    # find the knee angle
    knee = model.joints['knee_r']
    kneeAngle = model.joints['knee_r'].coordSets['knee_angle_r'].defaultValue
    knee_rTrans = np.zeros(3)

    # get the spline values
    trans1X = knee.getSimmSplineParams('translation1')[0]
    trans1Y = knee.getSimmSplineParams('translation1')[1]
    f = interp1d(trans1X, trans1Y, kind='cubic')

    knee_rTrans[0] = f(kneeAngle)

    trans2X = knee.getSimmSplineParams('translation2')[0]
    trans2Y = knee.getSimmSplineParams('translation2')[1]
    f2 = interp1d(trans2X, trans2Y, kind='cubic')
    knee_rTrans[1] = f2(kneeAngle)

    trans = None
    if body == 'pelvis':
        trans = np.zeros(3)
    elif body == 'femur_l':
        trans = model.joints['hip_l'].locationInParent
    elif body == 'femur_r':
        trans = model.joints['hip_r'].locationInParent
    elif body == 'tibia_l':
        trans = (model.joints['hip_l'].locationInParent +
                 knee_lTrans)
    elif body == 'tibia_r':
        trans = (model.joints['hip_r'].locationInParent +
                 knee_rTrans)
    elif body == 'talus_l':
        trans = (model.joints['hip_l'].locationInParent +
                 knee_lTrans +
                 model.joints['ankle_l'].locationInParent)
    elif body == 'talus_r':
        trans = (model.joints['hip_r'].locationInParent +
                 knee_rTrans +
                 model.joints['ankle_r'].locationInParent)
    elif body == 'calcn_l':
        trans = (model.joints['hip_l'].locationInParent +
                 knee_lTrans +
                 model.joints['ankle_l'].locationInParent +
                 model.joints['subtalar_l'].locationInParent)
    elif body == 'calcn_r':
        trans = (model.joints['hip_r'].locationInParent +
                 knee_rTrans +
                 model.joints['ankle_r'].locationInParent +
                 model.joints['subtalar_r'].locationInParent)
    elif body == 'toes_l':
        trans = (model.joints['hip_l'].locationInParent +
                 knee_lTrans +
                 model.joints['ankle_l'].locationInParent +
                 model.joints['subtalar_l'].locationInParent +
                 model.joints['mtp_l'].locationInParent)
    elif body == 'toes_r':
        trans = (model.joints['hip_r'].locationInParent +
                 knee_rTrans +
                 model.joints['ankle_r'].locationInParent +
                 model.joints['subtalar_r'].locationInParent +
                 model.joints['mtp_r'].locationInParent)

    return trans


class Gait2392MuscleCustomiser(object):

    def __init__(self, config, ll=None, osimmodel=None, landmarks=None):
        """
        Class for customising gait2392 muscle points using host-mesh fitting

        inputs
        ======
        config : dict
            Dictionary of option. (work in progress) Example:
            {
            'osim_output_dir': '/path/to/output/model.osim',
            'in_unit': 'mm',
            'out_unit': 'm',
            'write_osim_file': True,
            'update_knee_splines': False,
            'static_vas': False,
            }
        ll : LowerLimbAtlas instance
            Model of lower limb bone geometry and pose
        osimmodel : opensim.Model instance
            The opensim model instance to customise

        """
        self.config = config
        self.ll = ll
        self.trcdata = landmarks
        self.gias_osimmodel = None
        if osimmodel is not None:
            self.set_osim_model(osimmodel)
        self._unit_scaling = dim_unit_scaling(
            self.config['in_unit'], self.config['out_unit']
        )

    def set_osim_model(self, model):
        self.gias_osimmodel = osim.Model(model=model)

    def cust_pelvis(self):

        pelvis = self.ll.models['pelvis']

        # load the pelvis muscle attachment node numbers
        with open(DATA_DIR + 'pelvisNodeNumbers.txt') as infile:
            pelvisData = json.load(infile)

        pelvisAttachmentNodeNums = list(pelvisData.values())
        pelvisMuscleNames = list(pelvisData.keys())
        pelvisMuscleNames = [str(item) for item in pelvisMuscleNames]

        # This method appears to be taking quite a while to complete (like 5
        # minutes), is this expected? This wasn't being used in musclecusthfm.
        # the muscle attachments were selected an a 24x24 mesh
        pelvisPoints, lhF = pelvis.gf.triangulate([24, 24])

        # Align the discretised pelvis points and the muscle attachments to the
        # opensims pelvis local coordinate system.
        localPelvisPoints = pelvis.acs.map_local(pelvisPoints) / 1000
        pelvisAttachments = localPelvisPoints[pelvisAttachmentNodeNums]

        for i in range(len(pelvisMuscleNames)):
            muscle = self.gias_osimmodel.muscles[str(pelvisMuscleNames[i])]
            pathPoints = muscle.path_points
            s = sorted(muscle.path_points.keys())

            aSite = None
            # aSite will be 0 if attachment is an origin and -1 if insertion
            if pathPoints[s[0]].body.name == 'pelvis':
                aSite = 0
            elif pathPoints[s[-1]].body.name == 'pelvis':
                aSite = -1

            # update the location of the pathpoint
            pp = pathPoints[s[aSite]]
            pp.location = pelvisAttachments[i]

    def cust_femur_l(self):

        leftFemur = self.ll.models['femur-l']

        # load in the femur muscle attachment node numbers
        with open(DATA_DIR + 'leftFemurNodeNumbers.txt') as infile:
            leftFemurData = json.load(infile)

        leftFemurAttachmentNodeNums = list(leftFemurData.values())
        leftFemurMuscleNames = list(leftFemurData.keys())
        leftFemurMuscleNames = [str(item) for item in leftFemurMuscleNames]

        # update the geometric field coordinate system to match opensims
        update_femur_opensim_acs(leftFemur)

        # the muscle attachments were selected an a 24x24 mesh
        leftFemurPoints, lhF = leftFemur.gf.triangulate([24, 24])

        # align the discretised femur points and the muscle attachments to the
        # opensims femur local coordinate system
        localLeftFemurPoints = leftFemur.acs.map_local(leftFemurPoints) / 1000
        leftFemurAttachments = localLeftFemurPoints[
            leftFemurAttachmentNodeNums]

        for i in range(len(leftFemurMuscleNames)):
            muscleLeft = self.gias_osimmodel.muscles[
                str(leftFemurMuscleNames[i])]
            pathPointsLeft = muscleLeft.path_points
            sL = sorted(muscleLeft.path_points.keys())

            aSite = None
            # aSite will be 0 if attachment is an origin and -1 if insertion
            if pathPointsLeft[sL[0]].body.name == 'femur_l':
                aSite = 0
            elif pathPointsLeft[sL[-1]].body.name == 'femur_l':
                aSite = -1

            # update the location of the pathpoint
            ppL = pathPointsLeft[sL[aSite]]
            ppL.location = leftFemurAttachments[i]

    def cust_femur_r(self):

        rightFemur = self.ll.models['femur-r']
        rightFemur.side = 'right'

        with open(DATA_DIR + 'rightFemurNodeNumbers.txt') as infile:
            rightFemurData = json.load(infile)

        rightFemurAttachmentNodeNums = list(rightFemurData.values())
        rightFemurMuscleNames = list(rightFemurData.keys())
        rightFemurMuscleNames = [str(item) for item in rightFemurMuscleNames]

        # update the geometric field coordinate system to match opensims
        update_femur_opensim_acs(rightFemur)

        rightFemurPoints, rhF = rightFemur.gf.triangulate([24, 24])

        localRightFemurPoints = rightFemur.acs.map_local(
            rightFemurPoints) / 1000
        rightFemurAttachments = localRightFemurPoints[
            rightFemurAttachmentNodeNums]

        # update attachments
        for i in range(len(rightFemurMuscleNames)):
            muscleRight = self.gias_osimmodel.muscles[
                str(rightFemurMuscleNames[i])]
            pathPointsRight = muscleRight.path_points
            sR = sorted(muscleRight.path_points.keys())

            aSite = None
            # aSite will be 0 if attachment is an origin and -1 if insertion
            if pathPointsRight[sR[0]].body.name == 'femur_r':
                aSite = 0
            elif pathPointsRight[sR[-1]].body.name == 'femur_r':
                aSite = -1

            ppR = pathPointsRight[sR[aSite]]
            ppR.location = rightFemurAttachments[i]

    def cust_tibia_l(self):
        # The tibia, patella and fibula all use the same fieldwork model to
        # align with opensim

        leftTibFib = self.ll.models['tibiafibula-l']
        leftPatella = self.ll.models['patella-l']
        update_tibiafibula_opensim_acs(leftTibFib)

        leftTib, leftFib = split_tibia_fibula_gfs(leftTibFib.gf)

        # load in the tibia muscle attachment node numbers
        with open(DATA_DIR + 'leftTibiaNodeNumbers.txt') as infile:
            leftTibiaData = json.load(infile)

        leftTibiaAttachmentNodeNums = list(leftTibiaData.values())
        leftTibiaMuscleNames = list(leftTibiaData.keys())
        leftTibiaMuscleNames = [str(item) for item in leftTibiaMuscleNames]

        # load in the fibula muscle attachment node numbers
        with open(DATA_DIR + 'leftFibulaNodeNumbers.txt') as infile:
            leftFibulaData = json.load(infile)

        leftFibulaAttachmentNodeNums = list(leftFibulaData.values())
        leftFibulaMuscleNames = list(leftFibulaData.keys())
        leftFibulaMuscleNames = [str(item) for item in leftFibulaMuscleNames]

        # load in the patella muscle attachment node numbers
        with open(DATA_DIR + 'leftPatellaNodeNumbers.txt') as infile:
            leftPatellaData = json.load(infile)

        leftPatellaAttachmentNodeNums = list(leftPatellaData.values())
        leftPatellaMuscleNames = list(leftPatellaData.keys())
        leftPatellaMuscleNames = [str(item) for item in leftPatellaMuscleNames]

        leftTibiaPoints, lhF = leftTib.triangulate([24, 24])
        leftFibulaPoints, lhF = leftFib.triangulate([24, 24])
        leftPatellaPoints, lhf = leftPatella.gf.triangulate([24, 24])

        localLeftTibiaPoints = leftTibFib.acs.map_local(leftTibiaPoints) / 1000
        leftTibiaAttachments = localLeftTibiaPoints[
            leftTibiaAttachmentNodeNums]

        localLeftFibulaPoints = leftTibFib.acs.map_local(
            leftFibulaPoints) / 1000
        leftFibulaAttachments = localLeftFibulaPoints[
            leftFibulaAttachmentNodeNums]

        localLeftPatellaPoints = leftTibFib.acs.map_local(
            leftPatellaPoints) / 1000
        leftPatellaAttachments = localLeftPatellaPoints[
            leftPatellaAttachmentNodeNums]

        # update the tibia attachments
        for i in range(len(leftTibiaMuscleNames)):
            muscleLeft = self.gias_osimmodel.muscles[
                str(leftTibiaMuscleNames[i])]
            pathPointsLeft = muscleLeft.path_points
            sL = sorted(muscleLeft.path_points.keys())

            aSite = None
            # aSite will be 0 if attachment is an origin and -1 if insertion
            if pathPointsLeft[sL[0]].body.name == 'tibia_l':
                aSite = 0
            elif pathPointsLeft[sL[-1]].body.name == 'tibia_l':
                aSite = -1

            ppL = pathPointsLeft[sL[aSite]]
            ppL.location = leftTibiaAttachments[i]

        # update the fibula attachments
        for i in range(len(leftFibulaMuscleNames)):
            muscleLeft = self.gias_osimmodel.muscles[
                str(leftFibulaMuscleNames[i])]
            pathPointsLeft = muscleLeft.path_points
            sL = sorted(muscleLeft.path_points.keys())

            aSite = None
            # aSite will be 0 if attachment is an origin and -1 if insertion
            if pathPointsLeft[sL[0]].body.name == 'tibia_l':
                aSite = 0
            elif pathPointsLeft[sL[-1]].body.name == 'tibia_l':
                aSite = -1

            ppL = pathPointsLeft[sL[aSite]]
            ppL.location = leftFibulaAttachments[i]

        # update the patella attachments
        for i in range(len(leftPatellaMuscleNames)):
            muscleLeft = self.gias_osimmodel.muscles[
                str(leftPatellaMuscleNames[i])]
            pathPointsLeft = muscleLeft.path_points
            sL = sorted(muscleLeft.path_points.keys())

            aSite = None
            # aSite will be 0 if attachment is an origin and -1 if insertion
            if pathPointsLeft[sL[0]].body.name == 'tibia_l':
                aSite = 0
            elif pathPointsLeft[sL[-1]].body.name == 'tibia_l':
                aSite = -1

            ppL = pathPointsLeft[sL[aSite]]
            ppL.location = leftPatellaAttachments[i]

    def cust_tibia_r(self):

        rightTibFib = self.ll.models['tibiafibula-r']
        rightPatella = self.ll.models['patella-r']
        update_tibiafibula_opensim_acs(rightTibFib)

        rightTib, rightFib = split_tibia_fibula_gfs(rightTibFib.gf)

        # load in the tibia attachment node numbers
        with open(DATA_DIR + 'rightTibiaNodeNumbers.txt') as infile:
            rightTibiaData = json.load(infile)

        rightTibiaAttachmentNodeNums = list(rightTibiaData.values())
        rightTibiaMuscleNames = list(rightTibiaData.keys())
        rightTibiaMuscleNames = [str(item) for item in rightTibiaMuscleNames]

        # load in the fibula attachment node numbers
        with open(DATA_DIR + 'rightFibulaNodeNumbers.txt') as infile:
            rightFibulaData = json.load(infile)

        rightFibulaAttachmentNodeNums = list(rightFibulaData.values())
        rightFibulaMuscleNames = list(rightFibulaData.keys())
        rightFibulaMuscleNames = [str(item) for item in rightFibulaMuscleNames]

        # load in the patella attachment node numbers
        with open(DATA_DIR + 'rightPatellaNodeNumbers.txt') as infile:
            rightPatellaData = json.load(infile)

        rightPatellaAttachmentNodeNums = list(rightPatellaData.values())
        rightPatellaMuscleNames = list(rightPatellaData.keys())
        rightPatellaMuscleNames = [
            str(item) for item in rightPatellaMuscleNames]

        rightTibiaPoints, lhF = rightTib.triangulate([24, 24])
        rightFibulaPoints, lhF = rightFib.triangulate([24, 24])
        rightPatellaPoints, lhf = rightPatella.gf.triangulate([24, 24])

        localRightTibiaPoints = rightTibFib.acs.map_local(
            rightTibiaPoints) / 1000
        rightTibiaAttachments = localRightTibiaPoints[
            rightTibiaAttachmentNodeNums]

        localRightFibulaPoints = rightTibFib.acs.map_local(
            rightFibulaPoints) / 1000
        rightFibulaAttachments = localRightFibulaPoints[
            rightFibulaAttachmentNodeNums]

        localRightPatellaPoints = rightTibFib.acs.map_local(
            rightPatellaPoints) / 1000
        rightPatellaAttachments = localRightPatellaPoints[
            rightPatellaAttachmentNodeNums]

        for i in range(len(rightTibiaMuscleNames)):
            muscleRight = self.gias_osimmodel.muscles[
                str(rightTibiaMuscleNames[i])]
            pathPointsRight = muscleRight.path_points
            sR = sorted(muscleRight.path_points.keys())

            aSite = None
            # aSite will be 0 if attachment is an origin and -1 if insertion
            if pathPointsRight[sR[0]].body.name == 'tibia_r':
                aSite = 0
            elif pathPointsRight[sR[-1]].body.name == 'tibia_r':
                aSite = -1

            ppR = pathPointsRight[sR[aSite]]
            ppR.location = rightTibiaAttachments[i]

        for i in range(len(rightFibulaMuscleNames)):
            muscleRight = self.gias_osimmodel.muscles[
                str(rightFibulaMuscleNames[i])]
            pathPointsRight = muscleRight.path_points
            sR = sorted(muscleRight.path_points.keys())

            aSite = None
            # aSite will be 0 if attachment is an origin and -1 if insertion
            if pathPointsRight[sR[0]].body.name == 'tibia_r':
                aSite = 0
            elif pathPointsRight[sR[-1]].body.name == 'tibia_r':
                aSite = -1

            ppR = pathPointsRight[sR[aSite]]
            ppR.location = rightFibulaAttachments[i]

        for i in range(len(rightPatellaMuscleNames)):
            muscleRight = self.gias_osimmodel.muscles[
                str(rightPatellaMuscleNames[i])]
            pathPointsRight = muscleRight.path_points
            sR = sorted(muscleRight.path_points.keys())

            aSite = None
            # aSite will be 0 if attachment is an origin and -1 if insertion
            if pathPointsRight[sR[0]].body.name == 'tibia_r':
                aSite = 0
            elif pathPointsRight[sR[-1]].body.name == 'tibia_r':
                aSite = -1

            ppR = pathPointsRight[sR[aSite]]
            ppR.location = rightPatellaAttachments[i]

    def write_cust_osim_model(self):
        self.gias_osimmodel.save(
            os.path.join(str(self.config['osim_output_dir']), OSIM_FILENAME)
        )

    def customise(self):
        # Note: a number of PathPoints that were scaled in the previous plugin
        # are also being scaled here. Are both of these necessary?

        self.cust_pelvis()
        self.cust_femur_l()
        self.cust_tibia_l()
        self.cust_femur_r()
        self.cust_tibia_r()

        # What is being done in the following methods that wasn't in the
        # previous plugin or one of the cust (^) methods? They seem to be
        # updating the same values that were updated earlier.
        self.update_hip_muscles()
        self.update_knee_muscles()
        self.update_foot_muscles()
        self.update_wrap_points()
        # The Marker Set was quite comprehensively updated in the previous
        # plugin, is the following method really important? Or better than the
        # other one?
        self.update_marker_set()

        if self.config['update_max_iso_forces']:
            self.update_max_iso_forces()

        # Currently, none of the OFL and TSL values are being re-calculated
        # after updating the PathPoints. They have been scaled in the previous
        # plugin but could be done more accurately here.

        if self.config['write_osim_file']:
            self.write_cust_osim_model()

    # This method assumes the current max iso force is in mm and multiplies it
    # to get the value in cm. I'm not sure it should be doing this (or not like
    # this at least). It should depend on the plugin configuration, right?
    def update_max_iso_forces(self):

        osimModel = self.gias_osimmodel
        subjectHeight = float(self.config['subject_height'])
        subjectMass = float(self.config['subject_mass'])

        # calculate muscle volumes using Handsfield (2014)
        osimAbbr, muscleVolume = muscleVolumeCalculator(
            subjectHeight, subjectMass)

        # load Opensim model muscle set
        allMuscles = osimModel.get_muscles()

        allMusclesNames = list(range(allMuscles.getSize()))
        oldValue = np.zeros([allMuscles.getSize(), 1])
        optimalFibreLength = np.zeros([allMuscles.getSize(), 1])
        penAngleAtOptFibLength = np.zeros([allMuscles.getSize(), 1])

        for i in range(allMuscles.getSize()):
            allMusclesNames[i] = allMuscles.get(i).getName()
            oldValue[i] = allMuscles.get(i).getMaxIsometricForce()
            optimalFibreLength[i] = allMuscles.get(i).getOptimalFiberLength()
            penAngleAtOptFibLength[i] = np.rad2deg(
                allMuscles.get(i).getPennationAngleAtOptimalFiberLength())

        # convert opt. fibre length from [m] to [cm] to match volume units
        # [cm^3]
        # Shouldn't this (and the volume units) depend on the plugin config?
        optimalFibreLength *= 100

        allMusclesNamesCut = list(range(allMuscles.getSize()))
        for i in range(len(allMusclesNames)):
            # delete trailing '_r' or '_l'
            currMuscleName = allMusclesNames[i][0:-2]

            # split the name from any digit in its name and only keep the first
            # string.
            currMuscleName = re.split(r'(\d+)', currMuscleName)
            currMuscleName = currMuscleName[0]

            # store in cell
            allMusclesNamesCut[i] = currMuscleName

        # calculate ratio of old max isometric forces for
        # multiple-lines-of-action muscles.
        newAbsVolume = np.zeros([allMuscles.getSize(), 1])
        fracOfGroup = np.zeros([allMuscles.getSize(), 1])

        for i in range(allMuscles.getSize()):

            currMuscleName = allMusclesNamesCut[i]
            currIndex = [
                j for j, x in enumerate(osimAbbr) if x == currMuscleName]
            # currIndex = osimAbbr.index(currMuscleName)
            if currIndex:
                currValue = muscleVolume[currIndex]
                newAbsVolume[i] = currValue

            # The peroneus longus/brevis and the extensors (EDL, EHL) have to
            # be treated seperatly as they are represented as a combined muscle
            # group in Handsfield, 2014. The following method may not be the
            # best!
            if currMuscleName == 'per_brev' or currMuscleName == 'per_long':
                currMuscleNameIndex = np.array([0, 0])
                tmpIndex = [j for j, x in enumerate(
                    allMusclesNamesCut) if x == 'per_brev']
                currMuscleNameIndex[0] = tmpIndex[0]
                tmpIndex = [j for j, x in enumerate(
                    allMusclesNamesCut) if x == 'per_long']
                currMuscleNameIndex[1] = tmpIndex[0]

                currIndex = [j for j, x in enumerate(osimAbbr) if x == 'per_']
                currValue = muscleVolume[currIndex]
                newAbsVolume[i] = currValue

            elif currMuscleName == 'ext_dig' or currMuscleName == 'ext_hal':
                currMuscleNameIndex = np.array([0, 0])
                tmpIndex = [j for j, x in enumerate(
                    allMusclesNamesCut) if x == 'ext_dig']
                currMuscleNameIndex[0] = tmpIndex[0]
                tmpIndex = [j for j, x in enumerate(
                    allMusclesNamesCut) if x == 'ext_hal']
                currMuscleNameIndex[1] = tmpIndex[0]

                currIndex = [j for j, x in enumerate(osimAbbr) if x == 'ext_']
                currValue = muscleVolume[currIndex]
                newAbsVolume[i] = currValue
            else:
                # find all instances of each muscle
                currMuscleNameIndex = [j for j, x in enumerate(
                    allMusclesNamesCut) if x == currMuscleName]
                # only require half of the results as we only need muscles from
                # one side
                currMuscleNameIndex = currMuscleNameIndex[0:int(len(
                    currMuscleNameIndex) / 2)]

            # find how much of the total muscle volume this muscle contributes
            fracOfGroup[i] = oldValue[i] / sum(oldValue[currMuscleNameIndex])

        # calculate new maximal isometric muscle forces

        specificTension = 61  # N/cm^2 from Zajac 1989
        newVolume = fracOfGroup * newAbsVolume
        # maxIsoMuscleForce = specificTension * (newVolume/optimalFibreLength)
        # * np.cos(math.degrees(penAngleAtOptFibLength))

        # Update muscles of loaded model (in workspace only!), change model
        # name and print new osim file.
        maxIsoMuscleForce = np.zeros([allMuscles.getSize(), 1])
        for i in range(allMuscles.getSize()):
            maxIsoMuscleForce[i] = specificTension * (
                    newVolume[i] / optimalFibreLength[i]) * np.cos(
                math.radians(penAngleAtOptFibLength[i]))

            # only update, if new value is not zero. Else do not override the
            # original value.
            if maxIsoMuscleForce[i] != 0:
                allMuscles.get(i).setMaxIsometricForce(maxIsoMuscleForce[i][0])

    def update_hip_muscles(self):

        muscleNames = ['glut_max1_l', 'glut_max2_l', 'glut_max3_l', 'peri_l',
                       'iliacus_l', 'psoas_l', 'glut_max1_r', 'glut_max2_r',
                       'glut_max3_r', 'peri_r', 'psoas_r', 'iliacus_r']
        joint = 'hip'
        body = 'pelvis'
        # joint - the joint that the muscles cross (currently only works for
        # muscles that cross a single joint)
        # body - the body that the origins of the muscles are attached to

        # this has only been tested for muscles that cross the hip

        # load in the original model
        mO = osim.Model(TEMPLATE_OSIM_PATH)
        mO.init_system()

        # for each muscle
        for i in range(len(muscleNames)):

            # display the pathpoints for both muscles
            muscleO = mO.muscles[muscleNames[i]]
            muscle = self.gias_osimmodel.muscles[muscleNames[i]]

            side = muscle.name[-2:]

            # find the transformation between the two bodies the muscles are
            # attached to
            transO = mO.joints[joint + side].locationInParent
            trans = self.gias_osimmodel.joints[joint + side].locationInParent

            pathPointsO = copy.copy(muscleO.path_points)
            pathPoints = copy.copy(muscle.path_points)

            for j in range(len(pathPointsO)):
                if list(pathPointsO.values())[j].body.name == body:
                    list(pathPointsO.values())[j].location -= transO
                    list(pathPoints.values())[j].location -= trans

                # ################################################## #
                # ###############Transform Points################### #
                # ################################################## #

            # find the path point names for the origin and the insertion
            sortedKeys = sorted(muscle.path_points.keys())

            # the origin will be the first sorted key and the insertion last
            orig = sortedKeys[0]
            ins = sortedKeys[-1]

            # find vector between origins and insertions
            v1 = pathPoints[orig].location - pathPointsO[orig].location
            v2 = pathPoints[ins].location - pathPointsO[ins].location

            # the new points are going to be found by translating the points
            # based on a weighting mulitplied by these two vectors

            # the weighting will be how far along the muscle the point it

            # find the total muscle length
            segments = np.zeros([len(pathPointsO) - 1, 3])
            lengths = np.zeros(len(pathPointsO) - 1)

            for j in range(len(pathPointsO) - 1):
                segments[j] = pathPointsO[muscle.name + '-P' + str(
                    j + 2)].location - pathPointsO[
                    muscle.name + '-P' + str(j + 1)].location
                lengths[j] = np.linalg.norm(segments[j])

            Tl = np.sum(lengths)

            # Define the weighting function
            # for the points calculate the magnitude of the new vector and at
            # what angle

            for j in range(len(pathPointsO) - 2):
                # the second pathpoint will be the first via point
                p = pathPointsO[muscle.name + '-P' + str(j + 2)].location

                # find how far along the muscle the point is
                dl = np.sum(lengths[:j + 1])

                # create the new points by finding adding a weighted vector
                pNew = ((dl / Tl) * v2) + ((1 - dl / Tl) * v1) + p

                # update the opensim model
                muscle.path_points[muscle.name + '-P' + str(
                    j + 2)].location = pNew

            # tranform the points back to the main body local coordinate system
            for j in range(len(pathPoints)):

                if list(pathPoints.values())[j].body.name == body:
                    list(pathPoints.values())[j].location += trans

    def update_knee_muscles(self):

        muscleNames = ['bifemlh_l', 'semimem_l', 'semiten_l', 'sar_l', 'tfl_l',
                       'grac_l', 'rect_fem_l', 'bifemlh_r', 'semimem_r',
                       'semiten_r', 'sar_r', 'tfl_r', 'grac_r', 'rect_fem_r',
                       'bifemsh_l', 'vas_med_l', 'vas_int_l', 'vas_lat_l',
                       'bifemsh_r', 'vas_med_r', 'vas_int_r', 'vas_lat_r',
                       'med_gas_l', 'lat_gas_l', 'med_gas_r', 'lat_gas_r']

        # This is being done multiple times. Should move outside this method.
        # load in the original model
        mO = osim.Model(TEMPLATE_OSIM_PATH)
        mO.init_system()

        for i in range(len(muscleNames)):
            # display the pathpoints for both muscles
            muscleO = mO.muscles[muscleNames[i]]
            muscle = self.gias_osimmodel.muscles[muscleNames[i]]

            pathPointsO = copy.copy(muscleO.path_points)
            pathPoints = copy.copy(muscle.path_points)

            for j in range(len(pathPointsO)):
                list(pathPointsO.values())[j].location += local_osim_2_global(
                    list(pathPointsO.values())[j].body.name, mO)
                list(pathPoints.values())[j].location += local_osim_2_global(
                    list(pathPoints.values())[j].body.name,
                    self.gias_osimmodel)

            # find the path point names for the origin and the insertion
            sortedKeys = sorted(muscle.path_points.keys())

            # the origin will be the first sorted key and the insertion last
            orig = sortedKeys[0]
            ins = sortedKeys[-1]

            # find vector between origins and insertions
            v1 = pathPoints[orig].location - pathPointsO[orig].location
            v2 = pathPoints[ins].location - pathPointsO[ins].location

            # the new points are going to be found by translating the points
            # based on a weighting mulitplied by these two vectors

            # the weighting will be how far along the muscle the point it

            # find the total muscle length
            segments = np.zeros([len(pathPointsO) - 1, 3])
            lengths = np.zeros(len(pathPointsO) - 1)
            for j in range(len(pathPointsO) - 1):
                segments[j] = pathPointsO[muscle.name + '-P' + str(
                    j + 2)].location - pathPointsO[
                    muscle.name + '-P' + str(j + 1)].location
                lengths[j] = np.linalg.norm(segments[j])

            Tl = np.sum(lengths)

            # Define the weighting function for the points calculate the
            # magnitude of the new vector and at what angle

            for j in range(len(pathPointsO) - 2):
                # the second pathpoint will be the first via point
                p = pathPointsO[muscle.name + '-P' + str(j + 2)].location

                # find how far along the muscle the point is
                dl = np.sum(lengths[:j + 1])

                # create the new points by finding adding a weighted vector
                pNew = ((dl / Tl) * v2) + ((1 - dl / Tl) * v1) + p

                # update the opensim model
                muscle.path_points[muscle.name + '-P' + str(
                    j + 2)].location = pNew

            # tranform the pelvis points back to the pelvis region
            for j in range(len(pathPoints)):
                list(pathPoints.values())[j].location -= local_osim_2_global(
                    list(pathPoints.values())[j].body.name,
                    self.gias_osimmodel)

    def update_foot_muscles(self):

        muscleNames = ['ext_dig_l', 'ext_hal_l', 'flex_dig_l', 'flex_hal_l',
                       'per_brev_l', 'per_long_l', 'per_tert_l', 'tib_ant_l',
                       'tib_post_l', 'ext_dig_r', 'ext_hal_r', 'flex_dig_r',
                       'flex_hal_r', 'per_brev_r', 'per_long_r', 'per_tert_r',
                       'tib_ant_r', 'tib_post_r']

        # load in the original model
        mO = osim.Model(TEMPLATE_OSIM_PATH)
        mO.init_system()

        for i in range(len(muscleNames)):

            # get the pathPoints for the old and new muscle
            muscleO = mO.muscles[muscleNames[i]]
            muscle = self.gias_osimmodel.muscles[muscleNames[i]]

            side = muscle.name[-1]

            # find the transformation between the two bodies the muscles are
            # attached to
            transO = mO.joints['ankle_' + side].locationInParent + mO.joints[
                'subtalar_' + side].locationInParent
            trans = self.gias_osimmodel.joints['ankle_' + side]\
                        .locationInParent + self.gias_osimmodel.joints[
                'subtalar_' + side].locationInParent

            pathPointsO = copy.copy(muscleO.path_points)
            pathPoints = copy.copy(muscle.path_points)

            # ################################################## #
            # ###############Transform Points################### #
            # ################################################## #

            # find the path point names for the origin and the insertion
            sortedKeys = sorted(muscle.path_points.keys())

            # the origin will be the first sorted key
            orig = sortedKeys[0]

            ins = None
            # find the first point on the calcn
            for j in sortedKeys:
                if pathPoints[j].body.name == 'calcn_' + side:
                    ins = j
                    break

            endPP = sortedKeys.index(ins)

            for j in range(endPP + 1):

                if pathPointsO[sortedKeys[j]].body.name == 'calcn_' + side:
                    pathPointsO[sortedKeys[j]].location += transO
                    pathPoints[sortedKeys[j]].location += trans

                # find vector between origins and insertions
            v1 = pathPoints[orig].location - pathPointsO[orig].location
            v2 = pathPoints[ins].location - pathPointsO[ins].location

            # the new points are going to be found by translating the points
            # based on a weighting mulitplied by these two vectors

            # the weighting will be how far along the muscle the point it

            # find the total muscle length
            segments = np.zeros([endPP, 3])
            lengths = np.zeros(endPP)
            for j in range(endPP):
                segments[j] = pathPointsO[muscle.name + '-P' + str(
                    j + 2)].location - pathPointsO[
                    muscle.name + '-P' + str(j + 1)].location
                lengths[j] = np.linalg.norm(segments[j])

            Tl = np.sum(lengths)

            # Define the weighting function for the points calculate the
            # magnitude of the new vector and at what angle

            for j in range(endPP - 1):
                # the second pathpoint will be the first via point
                p = pathPointsO[muscle.name + '-P' + str(j + 2)].location

                # find how far along the muscle the point is
                dl = np.sum(lengths[:j + 1])

                # create the new points by finding adding a weighted vector
                pNew = ((dl / Tl) * v2) + ((1 - dl / Tl) * v1) + p

                # update the opensim model
                muscle.path_points[muscle.name + '-P' + str(
                    j + 2)].location = pNew

            for j in range(endPP + 1):
                if pathPoints[sortedKeys[j]].body.name == 'calcn_' + side:
                    pathPoints[sortedKeys[j]].location -= trans

    def update_wrap_points(self):

        muscleNames = ['psoas_l', 'iliacus_l', 'psoas_r', 'iliacus_r']
        wrapNames = ['PS_at_brim_l', 'IL_at_brim_l', 'PS_at_brim_r',
                     'IL_at_brim_r']
        joint = 'hip'
        wrapPoints = {'psoas_l': 26, 'psoas_r': 26, 'iliacus_l': 4926,
                      'iliacus_r': 26}

        for i in range(len(muscleNames)):

            wrap = self.gias_osimmodel.wrapObjects[wrapNames[i]]

            radiiString = wrap.getDimensions()

            # increase the radii by a small amount so the via point don't sit
            # directly on the wrap object
            radii = np.array(str.split(radiiString))[1:].astype(float) + 0.002

            theta = np.linspace(0, 2 * pi, 100)
            phi = np.linspace(0, pi, 50)
            sphere = np.zeros([1, 3])

            wrapCentre = wrap.get_translation()

            for j in range(len(theta)):
                for k in range(len(phi)):
                    x = wrapCentre[0] + radii[0] * np.cos(theta[j]) * np.sin(
                        phi[k])
                    y = wrapCentre[1] + radii[1] * np.sin(theta[j]) * np.sin(
                        phi[k])
                    z = wrapCentre[2] + radii[2] * np.cos(phi[k])

                    if i == 0 and j == 0:
                        sphere[i, :] = [x, y, z]
                    else:
                        sphere = np.vstack([sphere, [x, y, z]])

            # with the sphere created get the via point
            muscle = self.gias_osimmodel.muscles[muscleNames[i]]

            viaPoint = muscle.path_points[muscle.name + '-P2']

            # find the closest point on the sphere
            newPoint = sphere[wrapPoints[muscle.name]]

            # update the path point
            viaPoint.location = newPoint

            # check if P-3 is inside the wrap surface
            checkPoint = muscle.path_points[muscle.name + '-P3']

            # tranform to global coordinates

            side = muscleNames[i][-2:]

            # find the transformation between the two bodies the muscles are
            # attached to
            trans = self.gias_osimmodel.joints[joint + side].locationInParent

            # find the distance between the closest point on the sphere and the
            # centre
            dists = sphere - (checkPoint.location + trans)

            # normalize the distances to each point
            normDists = np.linalg.norm(dists, axis=1)

            nodeNum = np.argmin(normDists)

            np_wrap_centre = np.array(
                [wrapCentre[0], wrapCentre[1], wrapCentre[2]])
            d1 = np.linalg.norm(np_wrap_centre - sphere[nodeNum])

            # find the distance between the point and the centre of the sphere
            d2 = np.linalg.norm(np_wrap_centre - (checkPoint.location + trans))

            # If the distance d1 is larger than d2 move the point is inside the
            # sphere
            # and needs to be moved to the closest point on the sphere
            if d1 > d2:
                checkPoint.location = sphere[nodeNum] - trans

    def update_marker_set(self):

        # create dictionary linking landmarks to bodies based on the Cleveland
        # Marker Set
        fieldworkMarkers = {
            'pelvis': ['RASI', 'LASI', 'RPSI', 'LPSI', 'SACR', 'LHJC', 'RHJC'],
            'femur_l': ['LT1', 'LT2', 'LT3', 'LKNE', 'LKNM', 'LKJC'],
            'femur_r': ['RT1', 'RT2', 'RT3', 'RKNE', 'RKNM', 'RKJC'],
            'tibia_l': ['LS1', 'LS2', 'LS3', 'LANK', 'LANM', 'LAJC'],
            'tibia_r': ['RS1', 'RS2', 'RS3', 'RANK', 'RANM', 'RAJC'],
        }

        otherMarkers = {
            'torso': ['C7', 'T10', 'CLAV', 'STRN', 'BackExtra', 'LSHO', 'LTRI',
                      'LELB', 'LWRI', 'RSHO', 'RTRI', 'RELB', 'RWRI'],
            'calcn_l': ['LHEE'],
            'toes_l': ['LTOE'],
            'calcn_r': ['RHEE'],
            'toes_r': ['RTOE']
        }

        self.gias_osimmodel.init_system()

        # load in the geometric fields and update their coordinate systems to
        # align with opensim.
        # This may have already been done?
        pelvis = self.ll.models['pelvis']

        femur_l = self.ll.models['femur-l']
        update_femur_opensim_acs(femur_l)

        femur_r = self.ll.models['femur-r']
        femur_r.side = 'right'
        update_femur_opensim_acs(femur_r)

        tibia_l = self.ll.models['tibiafibula-l']
        update_tibiafibula_opensim_acs(tibia_l)

        tibia_r = self.ll.models['tibiafibula-r']
        tibia_r.side = 'right'
        update_tibiafibula_opensim_acs(tibia_r)

        markerSet = osim.opensim.MarkerSet()

        # for each body with a fieldwork model, map the markers to its body
        data = self.landmarks

        # for each marker
        for i in data:

            body = None

            # find what body the marker belongs to
            for j in fieldworkMarkers.keys():
                for k in range(len(fieldworkMarkers[j])):
                    if fieldworkMarkers[j][k] == i:
                        body = self.gias_osimmodel.bodies[j]

                        newMarker = osim.Marker(bodyname=j, offset=eval(
                            j).acs.map_local(np.array([data[fieldworkMarkers[
                                j][k]]])).flatten() / 1000)
                        newMarker.name = i
                        markerSet.adoptAndAppend(newMarker.get_osim_marker())
                        break

                    if body is not None:
                        break

                    # if the body has no fieldwork model check if it can be
                    # found in the extra dictionary
            if body is None:

                # import pdb
                # pdb.set_trace()

                for j in otherMarkers.keys():
                    for k in range(len(otherMarkers[j])):
                        if otherMarkers[j][k] == i:
                            body = j

                            if body == 'torso':
                                pointOnParent = pelvis.acs.map_local(
                                    np.array([data[i]])).flatten() / 1000
                                # find the difference in body coordinates
                                diff = self.gias_osimmodel.joints[
                                    'back'].locationInParent
                                markerPos = pointOnParent - diff
                                newMarker = osim.Marker(
                                    bodyname=body, offset=markerPos)
                                newMarker.name = i
                                markerSet.adoptAndAppend(
                                    newMarker.get_osim_marker())

                            elif body == 'calcn_l':
                                pointOnParent = tibia_l.acs.map_local(
                                    np.array([data[i]])).flatten() / 1000

                                # find the difference in body coordinates
                                diff = self.gias_osimmodel.joints[
                                           'ankle_l'].locationInParent + \
                                    self.gias_osimmodel.joints[
                                        'subtalar_l'].locationInParent
                                markerPos = pointOnParent - diff
                                newMarker = osim.Marker(
                                    bodyname=body, offset=markerPos)
                                newMarker.name = i
                                markerSet.adoptAndAppend(
                                    newMarker.get_osim_marker())

                            elif body == 'calcn_r':
                                pointOnParent = tibia_r.acs.map_local(
                                    np.array([data[i]])).flatten() / 1000

                                # find the difference in body coordinates
                                diff = self.gias_osimmodel.joints[
                                           'ankle_r'].locationInParent + \
                                    self.gias_osimmodel.joints[
                                        'subtalar_r'].locationInParent
                                markerPos = pointOnParent - diff
                                newMarker = osim.Marker(
                                    bodyname=body, offset=markerPos)
                                newMarker.name = i
                                markerSet.adoptAndAppend(
                                    newMarker.get_osim_marker())

                            elif body == 'toes_l':
                                pointOnParent = tibia_l.acs.map_local(
                                    np.array([data[i]])).flatten() / 1000

                                # find the difference in body coordinates
                                diff = self.gias_osimmodel.joints[
                                           'ankle_r'].locationInParent + \
                                    self.gias_osimmodel.joints[
                                           'subtalar_r'].locationInParent + \
                                    self.gias_osimmodel.joints[
                                           'mtp_l'].locationInParent
                                markerPos = pointOnParent - diff
                                newMarker = osim.Marker(
                                    bodyname=body, offset=markerPos)
                                newMarker.name = i
                                markerSet.adoptAndAppend(
                                    newMarker.get_osim_marker())

                            elif body == 'toes_r':
                                pointOnParent = tibia_r.acs.map_local(
                                    np.array([data[i]])).flatten() / 1000

                                # find the difference in body coordinates
                                diff = self.gias_osimmodel.joints[
                                           'ankle_r'].locationInParent + \
                                    self.gias_osimmodel.joints[
                                           'subtalar_r'].locationInParent + \
                                    self.gias_osimmodel.joints[
                                           'mtp_r'].locationInParent
                                markerPos = pointOnParent - diff
                                newMarker = osim.Marker(
                                    bodyname=body, offset=markerPos)
                                newMarker.name = i
                                markerSet.adoptAndAppend(
                                    newMarker.get_osim_marker())

            if body is None:
                print('{} can not be identified as a valid landmark'.
                      format(i))

        # update the marker set of the model
        self.gias_osimmodel.set_marker_set(markerSet)
