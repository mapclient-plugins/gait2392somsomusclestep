# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'configuredialog.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_ConfigureDialog(object):
    def setupUi(self, ConfigureDialog):
        if not ConfigureDialog.objectName():
            ConfigureDialog.setObjectName(u"ConfigureDialog")
        ConfigureDialog.resize(550, 303)
        self.gridLayout = QGridLayout(ConfigureDialog)
        self.gridLayout.setObjectName(u"gridLayout")
        self.configGroupBox = QGroupBox(ConfigureDialog)
        self.configGroupBox.setObjectName(u"configGroupBox")
        self.formLayout = QFormLayout(self.configGroupBox)
        self.formLayout.setObjectName(u"formLayout")
        self.formLayout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.label_identifier = QLabel(self.configGroupBox)
        self.label_identifier.setObjectName(u"label_identifier")

        self.formLayout.setWidget(0, QFormLayout.LabelRole, self.label_identifier)

        self.lineEdit_identifier = QLineEdit(self.configGroupBox)
        self.lineEdit_identifier.setObjectName(u"lineEdit_identifier")

        self.formLayout.setWidget(0, QFormLayout.FieldRole, self.lineEdit_identifier)

        self.label_input_unit = QLabel(self.configGroupBox)
        self.label_input_unit.setObjectName(u"label_input_unit")

        self.formLayout.setWidget(1, QFormLayout.LabelRole, self.label_input_unit)

        self.comboBox_in_unit = QComboBox(self.configGroupBox)
        self.comboBox_in_unit.setObjectName(u"comboBox_in_unit")

        self.formLayout.setWidget(1, QFormLayout.FieldRole, self.comboBox_in_unit)

        self.label_output_unit = QLabel(self.configGroupBox)
        self.label_output_unit.setObjectName(u"label_output_unit")

        self.formLayout.setWidget(2, QFormLayout.LabelRole, self.label_output_unit)

        self.comboBox_out_unit = QComboBox(self.configGroupBox)
        self.comboBox_out_unit.setObjectName(u"comboBox_out_unit")

        self.formLayout.setWidget(2, QFormLayout.FieldRole, self.comboBox_out_unit)

        self.label_write_osim_file = QLabel(self.configGroupBox)
        self.label_write_osim_file.setObjectName(u"label_write_osim_file")

        self.formLayout.setWidget(3, QFormLayout.LabelRole, self.label_write_osim_file)

        self.checkBox_write_osim_file = QCheckBox(self.configGroupBox)
        self.checkBox_write_osim_file.setObjectName(u"checkBox_write_osim_file")

        self.formLayout.setWidget(3, QFormLayout.FieldRole, self.checkBox_write_osim_file)

        self.label_update_knee_splines = QLabel(self.configGroupBox)
        self.label_update_knee_splines.setObjectName(u"label_update_knee_splines")

        self.formLayout.setWidget(4, QFormLayout.LabelRole, self.label_update_knee_splines)

        self.checkBox_update_knee_splines = QCheckBox(self.configGroupBox)
        self.checkBox_update_knee_splines.setObjectName(u"checkBox_update_knee_splines")

        self.formLayout.setWidget(4, QFormLayout.FieldRole, self.checkBox_update_knee_splines)

        self.label_static_vas = QLabel(self.configGroupBox)
        self.label_static_vas.setObjectName(u"label_static_vas")

        self.formLayout.setWidget(5, QFormLayout.LabelRole, self.label_static_vas)

        self.checkBox_static_vas = QCheckBox(self.configGroupBox)
        self.checkBox_static_vas.setObjectName(u"checkBox_static_vas")

        self.formLayout.setWidget(5, QFormLayout.FieldRole, self.checkBox_static_vas)

        self.label_update_max_iso_forces = QLabel(self.configGroupBox)
        self.label_update_max_iso_forces.setObjectName(u"label_update_max_iso_forces")

        self.formLayout.setWidget(6, QFormLayout.LabelRole, self.label_update_max_iso_forces)

        self.checkBox_update_max_iso_forces = QCheckBox(self.configGroupBox)
        self.checkBox_update_max_iso_forces.setObjectName(u"checkBox_update_max_iso_forces")

        self.formLayout.setWidget(6, QFormLayout.FieldRole, self.checkBox_update_max_iso_forces)

        self.label_subject_height = QLabel(self.configGroupBox)
        self.label_subject_height.setObjectName(u"label_subject_height")

        self.formLayout.setWidget(7, QFormLayout.LabelRole, self.label_subject_height)

        self.lineEdit_subject_height = QLineEdit(self.configGroupBox)
        self.lineEdit_subject_height.setObjectName(u"lineEdit_subject_height")

        self.formLayout.setWidget(7, QFormLayout.FieldRole, self.lineEdit_subject_height)

        self.label_subject_mass = QLabel(self.configGroupBox)
        self.label_subject_mass.setObjectName(u"label_subject_mass")

        self.formLayout.setWidget(8, QFormLayout.LabelRole, self.label_subject_mass)

        self.lineEdit_subject_mass = QLineEdit(self.configGroupBox)
        self.lineEdit_subject_mass.setObjectName(u"lineEdit_subject_mass")

        self.formLayout.setWidget(8, QFormLayout.FieldRole, self.lineEdit_subject_mass)

        self.label_output_dir = QLabel(self.configGroupBox)
        self.label_output_dir.setObjectName(u"label_output_dir")

        self.formLayout.setWidget(9, QFormLayout.LabelRole, self.label_output_dir)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.lineEdit_osim_output_dir = QLineEdit(self.configGroupBox)
        self.lineEdit_osim_output_dir.setObjectName(u"lineEdit_osim_output_dir")

        self.horizontalLayout.addWidget(self.lineEdit_osim_output_dir)

        self.pushButton_osim_output_dir = QPushButton(self.configGroupBox)
        self.pushButton_osim_output_dir.setObjectName(u"pushButton_osim_output_dir")

        self.horizontalLayout.addWidget(self.pushButton_osim_output_dir)


        self.formLayout.setLayout(9, QFormLayout.FieldRole, self.horizontalLayout)


        self.gridLayout.addWidget(self.configGroupBox, 0, 0, 1, 1)

        self.buttonBox = QDialogButtonBox(ConfigureDialog)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setOrientation(Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Cancel|QDialogButtonBox.Ok)

        self.gridLayout.addWidget(self.buttonBox, 1, 0, 1, 1)

        QWidget.setTabOrder(self.lineEdit_identifier, self.lineEdit_subject_height)
        QWidget.setTabOrder(self.lineEdit_subject_height, self.lineEdit_subject_mass)
        QWidget.setTabOrder(self.lineEdit_subject_mass, self.comboBox_in_unit)
        QWidget.setTabOrder(self.comboBox_in_unit, self.comboBox_out_unit)
        QWidget.setTabOrder(self.comboBox_out_unit, self.checkBox_write_osim_file)
        QWidget.setTabOrder(self.checkBox_write_osim_file, self.checkBox_static_vas)
        QWidget.setTabOrder(self.checkBox_static_vas, self.checkBox_update_max_iso_forces)
        QWidget.setTabOrder(self.checkBox_update_max_iso_forces, self.lineEdit_osim_output_dir)
        QWidget.setTabOrder(self.lineEdit_osim_output_dir, self.pushButton_osim_output_dir)
        QWidget.setTabOrder(self.pushButton_osim_output_dir, self.buttonBox)

        self.retranslateUi(ConfigureDialog)
        self.buttonBox.accepted.connect(ConfigureDialog.accept)
        self.buttonBox.rejected.connect(ConfigureDialog.reject)

        QMetaObject.connectSlotsByName(ConfigureDialog)
    # setupUi

    def retranslateUi(self, ConfigureDialog):
        ConfigureDialog.setWindowTitle(QCoreApplication.translate("ConfigureDialog", u"Configure Fieldwork Somso Muscle Step", None))
        self.configGroupBox.setTitle("")
        self.label_identifier.setText(QCoreApplication.translate("ConfigureDialog", u"identifier:  ", None))
        self.label_input_unit.setText(QCoreApplication.translate("ConfigureDialog", u"Input unit:", None))
        self.label_output_unit.setText(QCoreApplication.translate("ConfigureDialog", u"Output unit:", None))
        self.label_write_osim_file.setText(QCoreApplication.translate("ConfigureDialog", u"Write Osim file:", None))
        self.checkBox_write_osim_file.setText("")
        self.label_update_knee_splines.setText(QCoreApplication.translate("ConfigureDialog", u"Update Knee Splines:", None))
        self.checkBox_update_knee_splines.setText("")
        self.label_static_vas.setText(QCoreApplication.translate("ConfigureDialog", u"Static Vastus:", None))
        self.checkBox_static_vas.setText("")
        self.label_update_max_iso_forces.setText(QCoreApplication.translate("ConfigureDialog", u"Update Max Iso Forces:", None))
        self.checkBox_update_max_iso_forces.setText("")
        self.label_subject_height.setText(QCoreApplication.translate("ConfigureDialog", u"Subject Height (m):", None))
        self.label_subject_mass.setText(QCoreApplication.translate("ConfigureDialog", u"Subject Mass (kg):", None))
        self.label_output_dir.setText(QCoreApplication.translate("ConfigureDialog", u"Output folder:", None))
        self.pushButton_osim_output_dir.setText(QCoreApplication.translate("ConfigureDialog", u"...", None))
    # retranslateUi

