import io
import logging
import os
from collections import namedtuple
from copy import deepcopy
from dataclasses import dataclass

import networkx as nx
import numpy as np
import parmed as pm
import rdkit
from IPython.core.display import display
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdFMCS
from rdkit.Chem.Draw import IPythonConsole, rdMolDraw2D
from simtk import unit

from transformato.state import IntermediateStateFactory
from transformato.system import SystemStructure

logger = logging.getLogger(__name__)


@dataclass
class DummyRegion:
    mol_name: str
    match_termin_real_and_dummy_atoms: dict
    connected_dummy_regions: list
    tlc: str

    def return_connecting_real_atom(self, dummy_atoms:list):

        for real_atom in self.match_termin_real_and_dummy_atoms:
            for dummy_atom in self.match_termin_real_and_dummy_atoms[real_atom]:
                if dummy_atom in dummy_atoms:
                    logger.debug(f'Connecting real atom: {real_atom}')
                    return real_atom 

        logger.critical('No connecting real atom was found!')
        return None
    
@dataclass
class MutationDefinition:
    atoms_to_be_mutated: list
    common_core: list
    dummy_region: DummyRegion
    lambda_value_electrostatic : float
    lambda_value_vdw: float
    vdw_atom_idx: list
    steric_mutation_to_default: bool

    
class ProposeMutationRoute(object):

    def __init__(self,
                 s1: SystemStructure,
                 s2: SystemStructure,
                 ):
        """
        A class that proposes the mutation route between two molecules with a 
        common core (same atom types) based on two mols and generates the mutation 
        objects to perform the mutation on the psf objects.
        Parameters
        ----------
        mol1: Chem.Mol
        mol2: Chem.Mol
        """

        mol1_name: str = 'm1'
        mol2_name: str = 'm2'
        self.system: dict = {'system1': s1, 'system2': s2}
        self.mols: dict = {mol1_name: s1.mol, mol2_name: s2.mol}
        self.graphs: dict = {mol1_name: s1.graph, mol2_name: s2.graph}
        self.psfs: dict = {mol1_name: s1.waterbox_psf[f":{s1.tlc}"], mol2_name: s2.waterbox_psf[f":{s2.tlc}"]}
        self._substructure_match: dict = {mol1_name: [], mol2_name: []}
        self.removed_indeces: dict = {mol1_name: [], mol2_name: []}
        self.added_indeces: dict = {mol1_name: [], mol2_name: []}
        self.s1_tlc = s1.tlc
        self.s2_tlc = s2.tlc

        self.terminal_real_atom_cc1 = -1
        self.terminal_real_atom_cc2 = -1
        self.terminal_dummy_atom_cc1 = -1
        self.terminal_dummy_atom_cc2 = -1

        self.bondCompare = rdFMCS.BondCompare.CompareAny
        self.atomCompare = rdFMCS.AtomCompare.CompareElements
        self.maximizeBonds = True
        self.matchValences = False
        self.completeRingsOnly = False
        self.ringMatchesRingOnly = True

        self.dummy_region_cc1 = None
        self.dummy_region_cc2 = None


    def _match_terminal_real_and_dummy_atoms_for_mol1(self):
        """
        Matches the terminal real and dummy atoms and returns a dict with real atom idx as key and a set of dummy atoms that connect
        to this real atom as a set
        """
        return self._match_terminal_real_and_dummy_atoms(self.mols['m1'], self.terminal_real_atom_cc1, self.terminal_dummy_atom_cc1)

    def _match_terminal_real_and_dummy_atoms_for_mol2(self):
        """
        Matches the terminal real and dummy atoms and returns a dict with real atom idx as key and a set of dummy atoms that connect
        to this real atom as a set
        """
        return self._match_terminal_real_and_dummy_atoms(self.mols['m2'], self.terminal_real_atom_cc2, self.terminal_dummy_atom_cc2)

    @staticmethod
    def _match_terminal_real_and_dummy_atoms(mol, real_atoms_cc: list, dummy_atoms_cc: list) -> dict:
        """
        Matches the terminal real and dummy atoms and returns a dict with real atom idx as key and a set of dummy atoms that connect
        to this real atom as a set

        Parameters
        ----------
        mol : [Chem.Mol]
            The mol object with the real and dummy atoms
        real_atoms_cc : list
            list of real atom idx
        dummy_atoms_cc : list
            list of dummy atom idx

        Returns
        -------
        [type]
            [description]
        """

        from collections import defaultdict

        real_atom_match_dummy_atom = defaultdict(set)
        for real_atom_idx in real_atoms_cc:
            real_atom = mol.GetAtomWithIdx(real_atom_idx)
            real_neighbors = [x.GetIdx() for x in real_atom.GetNeighbors()]
            for dummy_atoms_idx in dummy_atoms_cc:
                if dummy_atoms_idx in real_neighbors:
                    real_atom_match_dummy_atom[real_atom_idx].add(dummy_atoms_idx)

        return real_atom_match_dummy_atom

    def _set_common_core_parameters(self):
        # find terminal atoms
        terminal_atoms_cc1, terminal_real_atoms_cc1 = self._find_terminal_atom(self.get_common_core_idx_mol1(), self.mols['m1'])
        terminal_atoms_cc2, terminal_real_atoms_cc2 = self._find_terminal_atom(self.get_common_core_idx_mol2(), self.mols['m2'])

        self.terminal_real_atom_cc1 = terminal_real_atoms_cc1
        self.terminal_real_atom_cc2 = terminal_real_atoms_cc2
        self.terminal_dummy_atom_cc1 = terminal_atoms_cc1
        self.terminal_dummy_atom_cc2 = terminal_atoms_cc2

        # match terminal real atoms between cc1 and cc2
        cc_idx_mol1 = self.get_common_core_idx_mol1()
        cc_idx_mol2 = self.get_common_core_idx_mol2()
        matching_terminal_atoms_betwee_cc = list()
        
        for cc1_idx, cc2_idx in zip(cc_idx_mol1, cc_idx_mol2):
            if cc1_idx in terminal_real_atoms_cc1 and cc2_idx in terminal_real_atoms_cc2:
                logger.info(f'Matching terminal atoms from cc1 to cc2. cc1: {cc1_idx} : cc2: {cc2_idx}') 
                matching_terminal_atoms_betwee_cc.append((cc1_idx, cc2_idx))
                
        if not matching_terminal_atoms_betwee_cc:
            raise RuntimeError('No terminal real atoms were matched between the common cores. Aborting.')
        
        self.matching_terminal_atoms_betwee_cc = matching_terminal_atoms_betwee_cc
         
    def calculate_common_core(self):

        # Calculate the MCS of m1 on m2
        self._find_mcs('m1', 'm2')
        # set the teriminal real/dummy atom indices
        self._set_common_core_parameters()
        # match the real/dummy atoms
        match_terminal_atoms_cc1 = self._match_terminal_real_and_dummy_atoms_for_mol1()
        # define connected dummy regions
        connected_dummy_regions_cc1 = self._find_connected_dummy_regions(
            mol_name='m1', 
            match_terminal_atoms_cc=match_terminal_atoms_cc1)
        
        self.dummy_region_cc1 = DummyRegion(mol_name='m1', 
                                            tlc=self.s1_tlc,
                                            match_termin_real_and_dummy_atoms=match_terminal_atoms_cc1, 
                                            connected_dummy_regions=connected_dummy_regions_cc1)

        match_terminal_atoms_cc2 = self._match_terminal_real_and_dummy_atoms_for_mol2()
        connected_dummy_regions_cc2 = self._find_connected_dummy_regions(
            mol_name='m2', 
            match_terminal_atoms_cc=match_terminal_atoms_cc2)
        
        self.dummy_region_cc2 = DummyRegion(mol_name='m2', 
                                            tlc=self.s2_tlc,                                           
                                            match_termin_real_and_dummy_atoms=match_terminal_atoms_cc2, 
                                            connected_dummy_regions=connected_dummy_regions_cc2)        
        
        # generate charge compmensated psfs
        psf1, psf2 = self._prepare_cc_for_charge_transfer()
        self.charge_compensated_ligand1_psf = psf1
        self.charge_compensated_ligand2_psf = psf2

    def _prepare_cc_for_charge_transfer(self):
        # we have to run the same charge mutation that will be run on cc2 to get the
        # charge distribution AFTER the full mutation

        # make a copy of the full psf
        m2_psf = self.psfs['m2'][:, :, :]
        m1_psf = self.psfs['m1'][:, :, :]
        charge_transformed_psfs = []

        for psf, tlc, cc_idx, dummy_region in zip([m1_psf, m2_psf],
                                    [self.s1_tlc, self.s2_tlc],
                                    [self.get_common_core_idx_mol1(), self.get_common_core_idx_mol2()],
                                    [self.dummy_region_cc1, self.dummy_region_cc2]):

            # set `initial_charge` parameter for Mutation
            for atom in psf.view[f":{tlc}"].atoms:
                # charge, epsilon and rmin are directly modiefied
                atom.initial_charge = atom.charge
    
            offset = min([atom.idx for atom in psf.view[f":{tlc}"].atoms])

            # getting copy of the atoms
            atoms_to_be_mutated = []
            for atom in psf.view[f":{tlc}"].atoms:
                idx = atom.idx - offset
                if idx not in cc_idx:
                    atoms_to_be_mutated.append(idx)

            logger.debug('############################')
            logger.debug('Preparing cc2 for charge transfer')
            logger.debug(f"Atoms for which charge is set to zero: {atoms_to_be_mutated}")
            logger.debug('############################')
            
            m = Mutation(tlc=tlc, atoms_to_be_mutated=atoms_to_be_mutated, common_core=cc_idx, dummy_region=dummy_region)
            m.mutate(psf, lambda_value_electrostatic=0.0)
            charge_transformed_psfs.append(psf)
        return charge_transformed_psfs[0], charge_transformed_psfs[1]

    def generate_mutation_list(self):

        # there are three obvious cases that we want to distinquish:
        # 1) mol1 is in mol2 (Methane -- Ethane)

        mutation_list = self.generate_mutations_to_common_core_for_mol1(
            nr_of_steps_for_electrostaticectrostatic=5, nr_of_steps_for_cc_transformation=2)
        # write intermediate states for systems
        intermediate_state = IntermediateStateFactory(
            system=self.system['system1'], mutation_list=mutation_list, configuration=configuration)
        intermediate_state.generate_intermediate_states()

        # generate mutation route
        mutation_list = self.generate_mutations_to_common_core_for_mol2(nr_of_steps_for_electrostaticectrostatic=5)
        # write intermediate states
        intermediate_state = IntermediateStateFactory(
            system=self.system['system2'], mutation_list=mutation_list, configuration=configuration)
        intermediate_state.generate_intermediate_states()

    def remove_idx_from_common_core_of_mol1(self, idx: int):
        self._remove_idx_from_common_core('m1', idx)

    def remove_idx_from_common_core_of_mol2(self, idx: int):
        self._remove_idx_from_common_core('m2', idx)

    def _remove_idx_from_common_core(self, name: str, idx: int):
        if idx in self.added_indeces[name] or idx in self._get_common_core(name):
            self.removed_indeces[name].append(idx)
            self._set_common_core_parameters()
        else:
            print(f"Idx: {idx} not in common core.")

    def add_idx_to_common_core_of_mol1(self, idx: int):
        self._add_common_core_atom('m1', idx)
        self._set_common_core_parameters()
        print(self.get_common_core_idx_mol1())

    def add_idx_to_common_core_of_mol2(self, idx: int):
        self._add_common_core_atom('m2', idx)
        self._set_common_core_parameters()
        print(self.get_common_core_idx_mol2())

    def _add_common_core_atom(self, name: str, idx: int):
        if idx in self.added_indeces[name] or idx in self._get_common_core(name):
            print(f"Idx: {idx} already in common core.")
            pass
        self.added_indeces[name].append(idx)

    def get_common_core_idx_mol1(self) -> list:
        """
        Returns the common core of mol1.
        """
        return self._get_common_core('m1')

    def get_common_core_idx_mol2(self) -> list:
        """
        Returns the common core of mol2.
        """
        return self._get_common_core('m2')

    def _get_common_core(self, name: str) -> list:
        """
        Helper Function - should not be called directly.
        Returns the common core.
        """
        keep_idx = []
        # BEWARE: the ordering is important - don't cast set!
        for idx in self._substructure_match[name] + self.added_indeces[name]:
            if idx not in self.removed_indeces[name]:
                keep_idx.append(idx)
        return keep_idx

    def _find_mcs(self,
                  mol1_name: str,
                  mol2_name: str,
                  maximizeBonds: bool = True,
                  matchValences: bool = False,
                  completeRingsOnly: bool = False,
                  ringMatchesRingOnly: bool = True

                  ):
        """
        A class that proposes the mutation route between two molecules with a 
        common core (same atom types) based on two mols and generates the mutation 
        objects to perform the mutation on the psf objects.
        Parameters
        ----------
        mol1_name: str
        mol2_name: str
        """

        logger.info('MCS starting ...')
        logger.info(f'bondCompare: {self.bondCompare}')
        logger.info(f'atomCompare: {self.atomCompare}')
        logger.info(f'maximizeBonds: {self.maximizeBonds}')
        logger.info(f'matchValences: {self.matchValences} ')
        logger.info(f'ringMatchesRingOnly: {self.ringMatchesRingOnly} ')
        logger.info(f'completeRingsOnly: {self.completeRingsOnly} ')

        m1, m2 = [deepcopy(self.mols[mol1_name]), deepcopy(self.mols[mol2_name])]

        for m in [m1, m2]:
            logger.info('Mol in SMILES format: {}.'.format(Chem.MolToSmiles(m, True)))

        # make copy of mols
        changed_mols = [Chem.Mol(x) for x in [m1, m2]]

        # find substructure match (ignore bond order but enforce element matching)
        mcs = rdFMCS.FindMCS(changed_mols,
                             bondCompare=self.bondCompare,
                             timeout=120,
                             atomCompare=self.atomCompare,
                             maximizeBonds=self.maximizeBonds,
                             matchValences=self.matchValences,
                             completeRingsOnly=self.completeRingsOnly,
                             ringMatchesRingOnly=self.ringMatchesRingOnly
                             )
        logger.info('Substructure match: {}'.format(mcs.smartsString))

        # convert from SMARTS
        mcsp = Chem.MolFromSmarts(mcs.smartsString, False)

        s1 = (m1.GetSubstructMatch(mcsp))
        logger.info('Substructere match idx: {}'.format(s1))
        self._display_mol(m1)
        s2 = (m2.GetSubstructMatch(mcsp))
        logger.info('Substructere match idx: {}'.format(s2))
        self._display_mol(m2)

        self._substructure_match[mol1_name] = list(s1)
        self._substructure_match[mol2_name] = list(s2)

    def _return_atom_idx_from_bond_idx(self, mol: Chem.Mol, bond_idx: int):
        return mol.GetBondWithIdx(bond_idx).GetBeginAtomIdx(), mol.GetBondWithIdx(bond_idx).GetEndAtomIdx()

    def _find_connected_dummy_regions(self, mol_name: str, match_terminal_atoms_cc: dict):

        from itertools import chain

        sub = self._substructure_match[mol_name]

        #############################
        # start
        #############################
        mol = self.mols[mol_name]
        # find all dummy atoms
        dummy_list_mol = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetIdx() not in sub]
        nr_of_dummy_atoms_mol = len(dummy_list_mol)
        # add all unique subgraphs here
        unique_subgraphs = []

        # iterate over dummy regions
        for real_atom in match_terminal_atoms_cc:
            logger.debug(f'real atom: {real_atom}')
            set_of_terminal_dummy_atoms = match_terminal_atoms_cc[real_atom]
            for terminal_dummy_atom in set_of_terminal_dummy_atoms:
                logger.debug(f'terminal_dummy_atom: {terminal_dummy_atom}')
                # start with biggest possible subgraph at final dummy atom
                for i in range(nr_of_dummy_atoms_mol, -1, -1):
                    subgraphs_of_length_i = []
                    logger.debug(f'Length: {i}')
                    all_subgraphs = Chem.FindUniqueSubgraphsOfLengthN(
                        mol=mol, length=i, useHs=True, useBO=False, rootedAtAtom=terminal_dummy_atom)
                    for subgraphs in all_subgraphs:
                        subgraphs = set(chain.from_iterable(
                            [self._return_atom_idx_from_bond_idx(mol=mol, bond_idx=e) for e in subgraphs]))
                        # test that only dummy atoms are in subgraph
                        if any([real_atom in subgraphs for real_atom in sub]):
                            pass
                        else:
                            subgraphs_of_length_i.append(set(subgraphs))

                    # test that new subgraphs are not already contained in bigger subgraphs
                    for subgraphs in subgraphs_of_length_i:
                        if any([subgraphs.issubset(old_subgraph) for old_subgraph in unique_subgraphs]):
                            # not new set
                            pass
                        else:
                            unique_subgraphs.append(set(subgraphs))
                            logger.debug(subgraphs)

        for dummy_atom in dummy_list_mol:
            if dummy_atom not in list(chain.from_iterable(unique_subgraphs)):
                unique_subgraphs.append(set([dummy_atom]))
        logger.debug(unique_subgraphs)
        return unique_subgraphs

    def _display_mol(self, mol: Chem.Mol):
        """
        Gets mol as input and displays its 2D Structure using IPythonConsole.
        Parameters
        ----------
        mol: Chem.Mol
            a rdkit mol object
        """

        def mol_with_atom_index(mol):
            atoms = mol.GetNumAtoms()
            for idx in range(atoms):
                mol.GetAtomWithIdx(idx).SetProp('molAtomMapNumber', str(mol.GetAtomWithIdx(idx).GetIdx()))
            return mol

        mol = mol_with_atom_index(mol)
        AllChem.Compute2DCoords(mol)
        display(mol)

    def show_common_core_on_mol1(self):
        """
        Shows common core on mol1        
        """
        return self._show_common_core(self.mols['m1'], self.get_common_core_idx_mol1())

    def show_common_core_on_mol2(self):
        """
        Shows common core on mol2        
        """
        return self._show_common_core(self.mols['m2'], self.get_common_core_idx_mol2())

    def _show_common_core(self, mol, highlight):
        """
        Helper function - do not call directly.
        Show common core.
        """
        # https://rdkit.blogspot.com/2015/02/new-drawing-code.html

        mol = deepcopy(mol)
        AllChem.Compute2DCoords(mol)

        drawer = rdMolDraw2D.MolDraw2DSVG(800, 800)
        drawer.SetFontSize(0.3)

        opts = drawer.drawOptions()

        for i in mol.GetAtoms():
            opts.atomLabels[i.GetIdx()] = str(i.GetProp('atom_index')) + ':' + i.GetProp('atom_type')

        drawer.DrawMolecule(mol, highlightAtoms=highlight)
        Draw.DrawingOptions.includeAtomNumbers = False
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText().replace('svg:', '')
        return(svg)

    def generate_mutations_to_common_core_for_mol1(self, nr_of_steps_for_electrostatic: int, nr_of_steps_for_cc_transformation: int) -> list:
        """
        Generates the mutation route to the common fore for mol1.
        Parameters
        ----------
        nr_of_steps_for_electrostatic : int
            nr of steps used for linearly scaling the charges to zero
        nr_of_steps_for_cc_transformation : int
        Returns
        ----------
        mutations: list
            list of mutations

        """
        if self.terminal_real_atom_cc1 == -1:
            raise RuntimeError('First generate the MCS. Aborting.')
        
      
        m = self._mutate_to_common_core('m1', self.dummy_region_cc1, self.get_common_core_idx_mol1(), nr_of_steps_for_electrostatic)
        t = self._transform_common_core(nr_of_steps_for_cc_transformation)

        return m + t

    def generate_mutations_to_common_core_for_mol2(self, nr_of_steps_for_electrostaticectrostatic: int) -> list:
        """
        Generates the mutation route to the common fore for mol2.
        Returns
        ----------
        mutations: list
            list of mutations        
        """
        if self.terminal_real_atom_cc1 == -1:
            raise RuntimeError('First generate the MCS')

        m = self._mutate_to_common_core('m2', self.get_common_core_idx_mol2(), nr_of_steps_for_electrostaticectrostatic)
        return m

    def _transform_common_core(self, nr_of_steps_for_cc_transformation: int) -> list:
        """
        Common Core 1 is transformed to Common core 2. Bonded parameters and charges are adjusted. 
        """

        transformations = []
        logger.info('##############################')
        logger.info('##############################')
        logger.info('Transform common core')
        logger.info('##############################')
        logger.info('##############################')

        # test if bonded mutations are necessary
        bonded_terms_mutation = False
        charge_mutation = False
        for cc1, cc2 in zip(self.get_common_core_idx_mol1() + [self.terminal_dummy_atom_cc1], self.get_common_core_idx_mol2() + [self.terminal_dummy_atom_cc2]):
            # did atom type change? if not don't add BondedMutations
            atom1 = self.psfs['m1'][cc1]
            print(atom1, atom1.type)
            atom2 = self.psfs['m2'][cc2]
            print(atom2, atom2.type)
            if atom1.type != atom2.type:
                logger.info('##############################')
                logger.info('Atom type transformation')
                logger.info(f'Atom that needs to be transformed: {atom1}.')
                logger.info(f'Atom type of atom in cc1: {atom1.type}.')
                logger.info(f'Template atom: {atom2}.')
                logger.info(f'Atom type of atom in cc2: {atom2.type}.')
                bonded_terms_mutation = True

        for cc1, cc2 in zip(self.get_common_core_idx_mol1(), self.get_common_core_idx_mol2()):
            atom1 = self.charge_compensated_ligand1_psf[cc1]
            atom2 = self.charge_compensated_ligand2_psf[cc2]
            if atom1.charge != atom2.charge:
                logger.info('##############################')
                logger.info('Charge transformation')
                logger.info('Charge needs to be transformed on common core')
                logger.info(f'Atom that needs to be transformed: {atom1}.')
                logger.info(f'Atom charge of atom in cc1: {atom1.charge}.')
                logger.info(f'Template atom: {atom2}.')
                logger.info(f'Atom charge of atom in cc2: {atom2.charge}.')
                charge_mutation = True

        # if necessary transform bonded parameters
        if bonded_terms_mutation or charge_mutation:
            logger.info(f'Bonded parameters mutation: {bonded_terms_mutation}.')
            logger.info(f'Charge parameters mutation: {charge_mutation}.')

            t = CommonCoreTransformation(
                self.get_common_core_idx_mol1(),
                self.get_common_core_idx_mol2(),
                self.psfs['m1'],
                self.psfs['m2'],
                nr_of_steps_for_cc_transformation,
                self.s1_tlc,
                self.s2_tlc,
                self.terminal_dummy_atom_cc1,
                self.terminal_dummy_atom_cc2,
                self.charge_compensated_ligand2_psf,
                charge_mutation=charge_mutation,
                bonded_terms_mutation=bonded_terms_mutation)
            transformations.append(t)
        else:
            logger.info('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            logger.info('No transformations needed.')
            logger.info('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            transformations = []

        return transformations

    @staticmethod
    def _find_terminal_atom(cc_idx: list, mol: Chem.Mol):
        """
        Find atoms that connect the  the molecule to the common core.

        Args:
            cc_idx (list): common core index atoms
            mol ([type]): rdkit mol object
        """
        terminal_dummy_atoms = []
        terminal_real_atoms = []

        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            if idx not in cc_idx:
                neighbors = [x.GetIdx() for x in atom.GetNeighbors()]
                if any([n in cc_idx for n in neighbors]):
                    terminal_dummy_atoms.append(idx)
            if idx in cc_idx:
                neighbors = [x.GetIdx() for x in atom.GetNeighbors()]
                if any([n not in cc_idx for n in neighbors]):
                    terminal_real_atoms.append(idx)

        logger.info(f"Terminal dummy atoms: {str(list(set(terminal_dummy_atoms)))}")
        logger.info(f"Terminal real atoms: {str(list(set(terminal_real_atoms)))}")

        return (list(set(terminal_dummy_atoms)), list(set(terminal_real_atoms)))

    # def _match_terminal_dummy_and_real_atoms

    def _mutate_to_common_core(self, dummy_region: DummyRegion, cc_idx: list, nr_of_steps_for_electrostatic: int) -> list:
        """
        Helper function - do not call directly.
        Generates the mutation route to the common fore for mol.
        """
        
        name = dummy_region.mol_name
        tlc = dummy_region.tlc
        
        mol = self.mols[name]
        charge_mutations = []
        lj_mutations = []
        
        # get the atom that connects the common core to the dummy regiom
        match_termin_real_and_dummy_atoms = dummy_region.match_termin_real_and_dummy_atoms

        # iterate through atoms and select atoms that need to be mutated
        atoms_to_be_mutated = []
        hydrogens = []
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            if idx not in cc_idx:
                # hydrogens are collected seperatly IF they are not terminal dummy atoms
                if atom.GetSymbol() == 'H' and idx not in match_termin_real_and_dummy_atoms.values():
                    hydrogens.append(idx)
                atoms_to_be_mutated.append(idx)
                logger.info('Will be decoupled: Idx:{} Element:{}'.format(idx, atom.GetSymbol()))


        if atoms_to_be_mutated:
            ############################################
            ############################################
            # charge mutation
            ############################################
            ############################################
            
            for step in range(0, nr_of_steps_for_electrostatic):
                lambda_value_electrostatic = 1- ((1/(nr_of_steps_for_electrostatic - 1)) * step) # NOTE nr_of_steps_for_electrostatic - 1!
                print(lambda_value_electrostatic)
                m = MutationDefinition(atoms_to_be_mutated=atoms_to_be_mutated, 
                                   common_core=cc_idx, 
                                   dummy_region=dummy_region,
                                   lambda_value_electrostatic = lambda_value_electrostatic,
                                   lambda_value_vdw = 1.0,
                                   vdw_atom_idx = [],
                                   steric_mutation_to_default = False)
            
                charge_mutations.append(m)
                
            ############################################
            ############################################
            # LJ mutation
            ############################################
            ############################################
            # save the last mutation steps
            lj_terminal_mutations = []

            # start with mutation of LJ of hydrogens
            # Only take hydrogens that are not terminal hydrogens
            if hydrogens:
                m = MutationDefinition(atoms_to_be_mutated=atoms_to_be_mutated, 
                                        common_core=cc_idx, 
                                        dummy_region=dummy_region,
                                        lambda_value_electrostatic = 1.0,
                                        lambda_value_vdw = 0.0,
                                        vdw_atom_idx = hydrogens,
                                        steric_mutation_to_default = False)

                lj_mutations.append(m)
            already_mutated = []
            # continue with scaling of heavy atoms LJ
            all_bonds = []

            # get all bonds
            for bond in nx.dfs_edges(self.graphs[name]):
                logger.debug(bond)
                all_bonds.append(bond)

            for idx1, idx2 in all_bonds:
                # continue if atom is not a hydrogen/already mutated and in the list of to be mutated atoms
                if idx1 in atoms_to_be_mutated and idx1 not in hydrogens and idx1 not in already_mutated:
                    # is it a terminal atom?
                    if idx1 in terminal_dummy_atoms:
                        lj_terminal_mutations.append(StericToDefaultMutation([idx1]))
                    else:
                        lj_mutations.append(StericToZeroMutation([idx1]))
                    already_mutated.append(idx1)
                # continue if atom is not a hydrogen/already mutated and in the list of to be mutated atoms
                if idx2 in atoms_to_be_mutated and idx2 not in hydrogens and idx2 not in already_mutated:
                    if idx2 in terminal_dummy_atoms:
                        lj_terminal_mutations.append(StericToDefaultMutation([idx2]))
                    else:
                        lj_mutations.append(StericToZeroMutation([idx2]))
                    already_mutated.append(idx2)

            # test that all mutations are included
            # TODO: test that all mutations are covered
            mutations = charge_mutations + lj_mutations + lj_terminal_mutations

            for m in mutations:
                if type(m) == ChargeMutation:
                    logger.debug(f"charge mutation on: {str(m.atom_idx)}")
                elif type(m) == StericMutation:
                    logger.debug(f"steric mutation on: {str(m.atom_idx)}")
                else:
                    logger.debug(f"mutation on: {str(m.atom_idx)}")
        else:
            logger.info("No atoms will be decoupled.")
            mutations = []
        return mutations


class CommonCoreTransformation(object):

    def __init__(self,
                 cc1_indicies: list,
                 cc2_indicies: list,
                 ligand1_psf: pm.charmm.CharmmPsfFile,
                 ligand2_psf: pm.charmm.CharmmPsfFile,
                 nr_of_steps: int,
                 tlc_cc1: str,
                 tlc_cc2: str,
                 terminal_atom_idx_cc1: int,
                 terminal_atom_idx_cc2: int,
                 charge_compensated_ligand2_psf: pm.charmm.CharmmPsfFile,
                 charge_mutation: bool,
                 bonded_terms_mutation: bool
                 ):
        """
        Scale the bonded parameters inside the common core.
        Parameters
        ----------
        cc1_indicies : list
            indices of cc1
        cc2_indicies : list
            indices of cc2 (in the same order as cc1)
        ligand1_psf : pm.charmm.CharmmPsfFile (copy of only ligand)
        ligand2_psf : pm.charmm.CharmmPsfFile (copy of only ligand)
            the target psf that is used to generate the new bonded parmaeters
        nr_of_steps : int
        tlc_cc1 : str
            three letter code of ligand in cc1
        tlc_cc2 : str
            three letter code of ligand in cc2
        """
        self.cc1_indicies = cc1_indicies
        self.cc2_indicies = cc2_indicies
        self.ligand2_psf = ligand2_psf
        self.ligand1_psf = ligand1_psf
        self.nr_of_steps = nr_of_steps
        assert(self.nr_of_steps >= 2)
        self.tlc_cc1 = tlc_cc1
        self.tlc_cc2 = tlc_cc2
        self.terminal_atom_idx_cc1 = terminal_atom_idx_cc1
        self.terminal_atom_idx_cc2 = terminal_atom_idx_cc2
        self.atom_names_mapping, self.terminal_names_mapping = self._get_atom_mapping()
        self.atom_names_mapping_for_bonded_terms = {**self.atom_names_mapping, **self.terminal_names_mapping}
        self.charge_mutation = charge_mutation
        self.bonded_terms_mutation = bonded_terms_mutation
        self.charge_compensated_ligand2_psf = charge_compensated_ligand2_psf

        logger.info(f'Bonded terms mutation: {bonded_terms_mutation}')
        logger.info(f'Charge mutation: {charge_mutation}')

    def _get_atom_mapping(self):
        """
        _get_atom_mapping -- match the atom names of the common cores

        Returns
        -------
        [dict]
            matched common core atom names
        """
        # match atomes in common cores
        match_atom_names_cc1_to_cc2 = {}
        for cc1_idx, cc2_idx in zip(self.cc1_indicies, self.cc2_indicies):
            ligand1_atom = self.ligand1_psf[cc1_idx]
            ligand2_atom = self.ligand2_psf[cc2_idx]
            match_atom_names_cc1_to_cc2[ligand1_atom.name] = ligand2_atom.name

        # match terminal atoms
        match_terminal_atoms_cc1_to_cc2 = {
            self.ligand1_psf[self.terminal_atom_idx_cc1].name: self.ligand2_psf[self.terminal_atom_idx_cc2].name}

        return match_atom_names_cc1_to_cc2, match_terminal_atoms_cc1_to_cc2

    def _mutate_charges(self, psf: pm.charmm.CharmmPsfFile, tlc: str, scale: float):

        # common core of psf 1 is transformed to psf 2
        for ligand1_atom in psf.view[f":{tlc}"]:
            if ligand1_atom.name not in self.atom_names_mapping:
                continue
            found = False

            # compare to charge compenstated psf 2
            for ligand2_atom in self.charge_compensated_ligand2_psf:
                if self.atom_names_mapping[ligand1_atom.name] == ligand2_atom.name:
                    found = True
                    # are the atoms different?
                    logger.debug(f"Modifying atom: {ligand1_atom}")
                    logger.debug(f"Template atom: {ligand2_atom}")

                    # scale epsilon
                    logger.debug(f"Real charge: {ligand1_atom.charge}")
                    modified_charge = (1.0 - scale) * ligand1_atom.initial_charge + scale * ligand2_atom.charge
                    logger.debug(f"New epsilon: {modified_charge}")
                    ligand1_atom.charge = modified_charge

            if not found:
                raise RuntimeError('No corresponding atom in cc2 found')

    def _mutate_atoms(self, psf: pm.charmm.CharmmPsfFile, tlc: str, scale: float):
        """
        mutate atom types. 

        Raises
        ------
        RuntimeError
            if common core atoms can not be matched
        """
        # what will be changed
        mod_type = namedtuple('Atom', 'epsilon, rmin')
        logger.debug('#######################')
        logger.debug('mutate_atoms')

        # iterate through the atoms of the ligand of system1
        for ligand1_atom in psf.view[f":{tlc}"]:
            # continue if not in atom_names_mapping
            if ligand1_atom.name not in self.atom_names_mapping:
                continue

            found = False
            # iterate through the atoms the ligand of system2
            for ligand2_atom in self.ligand2_psf:
                # is there a match up?
                if self.atom_names_mapping[ligand1_atom.name] == ligand2_atom.name:
                    found = True
                    # are the atoms different?
                    if ligand1_atom.type != ligand2_atom.type:
                        self._modify_type(ligand1_atom, psf)
                        logger.debug(f"Modifying atom: {ligand1_atom}")
                        logger.debug(f"Template atom: {ligand2_atom}")

                        # scale epsilon
                        logger.debug(f"Real epsilon: {ligand1_atom.epsilon}")
                        modified_epsilon = (1.0 - scale) * ligand1_atom.epsilon + scale * ligand2_atom.epsilon
                        logger.debug(f"New epsilon: {modified_epsilon}")

                        # scale rmin
                        logger.debug(f"Real rmin: {ligand1_atom.rmin}")
                        modified_rmin = (1.0 - scale) * ligand1_atom.rmin + scale * ligand2_atom.rmin
                        logger.debug(f"New rmin: {modified_rmin}")

                        ligand1_atom.mod_type = mod_type(modified_epsilon, modified_rmin)

            if not found:
                raise RuntimeError('No corresponding atom in cc2 found')

    def _mutate_bonds(self, psf: pm.charmm.CharmmPsfFile, tlc: str, scale: float):

        logger.debug('#######################')
        logger.debug('mutate_bonds')

        mod_type = namedtuple('Bond', 'k, req')
        for ligand1_bond in psf.view[f":{tlc}"].bonds:

            ligand1_atom1_name = ligand1_bond.atom1.name
            ligand1_atom2_name = ligand1_bond.atom2.name
            # all atoms of the bond must be in cc
            # everything outside the cc are bonded terms between dummies or
            # between real atoms and dummies and we can ignore them for now
            if not all(elem in self.atom_names_mapping_for_bonded_terms for elem in [ligand1_atom1_name, ligand1_atom2_name]):
                continue

            found = False
            for ligand2_bond in self.ligand2_psf.bonds:
                ligand2_atom1_name = ligand2_bond.atom1.name
                ligand2_atom2_name = ligand2_bond.atom2.name
                # all atoms of the bond must be in cc
                if not all(elem in self.atom_names_mapping_for_bonded_terms.values() for elem in [ligand2_atom1_name, ligand2_atom2_name]):
                    continue

                # match the two bonds
                if sorted([self.atom_names_mapping_for_bonded_terms[e] for e in [ligand1_atom1_name, ligand1_atom2_name]]) == sorted([ligand2_atom1_name, ligand2_atom2_name]):
                    found = True
                    # are the bonds different?
                    if sorted([ligand1_bond.atom1.type, ligand1_bond.atom2.type]) == sorted([ligand2_bond.atom1.type, ligand2_bond.atom2.type]):
                        continue
                    logger.debug(f"Modifying bond: {ligand1_bond}")

                    logger.debug(f"Template bond: {ligand2_bond}")
                    logger.debug(f'Original value for k: {ligand1_bond.type.k}')
                    logger.debug(f"Target k: {ligand2_bond.type.k}")
                    new_k = ((1.0 - scale) * ligand1_bond.type.k) + (scale * ligand2_bond.type.k)
                    logger.debug(new_k)

                    modified_k = new_k

                    logger.debug(f"New k: {modified_k}")

                    logger.debug(f"Old req: {ligand1_bond.type.req}")
                    modified_req = ((1.0 - scale) * ligand1_bond.type.req) + (scale * ligand2_bond.type.req)
                    logger.debug(f"Modified bond: {ligand1_bond}")

                    ligand1_bond.mod_type = mod_type(modified_k, modified_req)
                    logger.debug(ligand1_bond.mod_type)

            if not found:
                logger.critical(ligand1_bond)
                raise RuntimeError('No corresponding bond in cc2 found: {}'.format(ligand1_bond))

    def _mutate_angles(self, psf: pm.charmm.CharmmPsfFile, tlc: str, scale: float):

        mod_type = namedtuple('Angle', 'k, theteq')
        for cc1_angle in psf.view[f":{tlc}"].angles:
            ligand1_atom1_name = cc1_angle.atom1.name
            ligand1_atom2_name = cc1_angle.atom2.name
            cc1_a3 = cc1_angle.atom3.name

            # only angles in cc
            if not all(elem in self.atom_names_mapping_for_bonded_terms for elem in [ligand1_atom1_name, ligand1_atom2_name, cc1_a3]):
                continue

            found = False
            for cc2_angle in self.ligand2_psf.angles:
                ligand2_atom1_name = cc2_angle.atom1.name
                ligand2_atom2_name = cc2_angle.atom2.name
                cc2_a3 = cc2_angle.atom3.name
                # only angles in cc
                if not all(elem in self.atom_names_mapping_for_bonded_terms.values() for elem in [ligand2_atom1_name, ligand2_atom2_name, cc2_a3]):
                    continue

                if sorted([self.atom_names_mapping_for_bonded_terms[e] for e in [ligand1_atom1_name, ligand1_atom2_name, cc1_a3]]) == sorted([ligand2_atom1_name, ligand2_atom2_name, cc2_a3]):
                    found = True
                    if sorted([cc1_angle.atom1.type, cc1_angle.atom2.type, cc1_angle.atom3.type]) == \
                            sorted([cc2_angle.atom1.type, cc2_angle.atom2.type, cc2_angle.atom3.type]):
                        continue

                    logger.debug(f"Modifying angle: {cc1_angle}")
                    logger.debug(f"Template bond: {cc2_angle}")
                    logger.debug('Scaling k and theteq')

                    logger.debug(f"Old k: {cc1_angle.type.k}")
                    modified_k = (1.0 - scale) * cc1_angle.type.k + scale * cc2_angle.type.k
                    logger.debug(f"New k: {modified_k}")

                    logger.debug(f"Old k: {cc1_angle.type.theteq}")
                    modified_theteq = (1.0 - scale) * cc1_angle.type.theteq + scale * cc2_angle.type.theteq
                    logging.debug(f"New k: {modified_theteq}")

                    cc1_angle.mod_type = mod_type(modified_k, modified_theteq)

            if not found:
                logger.critical(cc1_angle)
                raise RuntimeError('No corresponding angle in cc2 found')

    def _mutate_torsions(self, psf: pm.charmm.CharmmPsfFile, tlc: str, scale: float):

        mod_type = namedtuple('Torsion', 'phi_k, per, phase, scee, scnb')

        # get all torsions present in initial topology
        for cc1_torsion in psf.view[f":{tlc}"].dihedrals:
            ligand1_atom1_name = cc1_torsion.atom1.name
            ligand1_atom2_name = cc1_torsion.atom2.name
            cc1_a3 = cc1_torsion.atom3.name
            cc1_a4 = cc1_torsion.atom4.name
            # all atoms must be in the cc
            if not all(elem in self.atom_names_mapping_for_bonded_terms for elem in [ligand1_atom1_name, ligand1_atom2_name, cc1_a3, cc1_a4]):
                continue

            # get corresponding torsion types in the new topology
            for cc2_torsion in self.ligand2_psf.dihedrals:
                ligand2_atom1_name = cc2_torsion.atom1.name
                ligand2_atom2_name = cc2_torsion.atom2.name
                cc2_a3 = cc2_torsion.atom3.name
                cc2_a4 = cc2_torsion.atom4.name
                # only torsion in cc
                if not all(elem in self.atom_names_mapping_for_bonded_terms.values() for elem in [ligand2_atom1_name, ligand2_atom2_name, cc2_a3, cc2_a4]):
                    continue

                if sorted([self.atom_names_mapping_for_bonded_terms[e] for e in [ligand1_atom1_name, ligand1_atom2_name, cc1_a3, cc1_a4]]) == sorted([ligand2_atom1_name, ligand2_atom2_name, cc2_a3, cc2_a4]):
                    found = True
                    if sorted([cc1_torsion.atom1.type, cc1_torsion.atom2.type, cc1_torsion.atom3.type, cc1_torsion.atom3.type]) == \
                            sorted([cc2_torsion.atom1.type, cc2_torsion.atom2.type, cc2_torsion.atom3.type, cc2_torsion.atom4.type]):
                        continue

                    mod_types = []
                    if scale <= 0.5:
                        # torsion present at cc1 needs to be turned fully off starting from self.nr_of_steps/2
                        for torsion_t in cc1_torsion.type:
                            modified_phi_k = torsion_t.phi_k * max(((1.0 - scale * 2)), 0.0)
                            mod_types.append(mod_type(modified_phi_k, torsion_t.per, torsion_t.phase,
                                                      torsion_t.scee, torsion_t.scnb))
                    else:
                        # torsion present at cc1 needs to be turned fully off starting from self.nr_of_steps/2
                        for torsion_t in cc2_torsion.type:
                            modified_phi_k = torsion_t.phi_k * max((scale - 0.5) * 2, 0.0)
                            mod_types.append(mod_type(modified_phi_k, torsion_t.per, torsion_t.phase,
                                                      torsion_t.scee, torsion_t.scnb))

                    cc1_torsion.mod_type = mod_types
            if not found:
                logger.critical(cc1_torsion)
                raise RuntimeError('No corresponding torsion in cc2 found')

    def mutate(self, psf: pm.charmm.CharmmPsfFile, tlc: str, current_step: int, verbose: int = 0):
        """
        Mutates the bonded parameters of cc1 to cc2.
        Parameters
        ----------
        psf : pm.charmm.CharmmPsfFile
            psf that gets mutated
        tlc : str
        current_step : int
            the current step in the mutation protocoll
        only_charge : bool
            only charge is scaled from cc1 to cc2
        """

        assert(type(psf) == pm.charmm.CharmmPsfFile)
        scale = current_step / (self.nr_of_steps)
        if self.charge_mutation:
            logger.info(f" -- Charge parameters from cc1 are transformed to cc2.")
            logger.info(f"Scaling factor:{scale}")
            # scale charge
            self._mutate_charges(psf, tlc, scale)
        elif self.bonded_terms_mutation:
            logger.info(f" -- Atom/Bond/Angle/Torsion parameters from cc1 are transformed to cc2.")
            logger.info(f"Scaling factor:{scale}")
            # scale atoms
            self._mutate_atoms(psf, tlc, scale)
            # scale bonds
            self._mutate_bonds(psf, tlc, scale)
            # scale angles
            self._mutate_angles(psf, tlc, scale)
            # scale torsions
            self._mutate_torsions(psf, tlc, scale)
        else:
            logger.critical('Nothing to do. Is there someting wrong?')

    def _modify_type(self, atom: pm.Atom, psf: pm.charmm.CharmmPsfFile):

        if (hasattr(atom, 'initial_type')):
            # only change parameters
            pass
        else:
            logger.info(f"Setting RRR atomtype for atom: {atom}.")
            atom.type = f"RRR{psf.number_of_dummys}"
            atom.initial_type = atom.type
            psf.number_of_dummys += 1


class Mutation(object):

    def __init__(self, 
                 atoms_to_be_mutated: list,
                 common_core: list,
                 dummy_region: DummyRegion):

        assert(type(atoms_to_be_mutated) == list)
        self.atoms_to_be_mutated = atoms_to_be_mutated
        self.dummy_region = dummy_region
        self.tlc = dummy_region.tlc

    def _mutate_charge(self,
                       psf: pm.charmm.CharmmPsfFile,
                       lambda_value: float,
                       offset: int
                       ):

        total_charge = int(round(sum([atom.initial_charge for atom in psf.view[f":{self.tlc}"].atoms])))
        # scale the charge of all atoms
        for idx in self.atoms_to_be_mutated:
            odx = idx + offset
            atom = psf[odx]
            logger.debug(f"Scale charge on {atom}")
            logger.debug(f"Old charge: {atom.charge}")
            new_charge = float(np.round(atom.initial_charge * lambda_value, 4))
            atom.charge = new_charge
            logger.debug(f"New charge: {atom.charge}")

        if lambda_value != 1:
            # compensate for the total change in charge the terminal atom
            self._compensate_charge(psf, total_charge, offset)

    def _mutate_vdw(self,
                    psf: pm.charmm.CharmmPsfFile,
                    lambda_value: float,
                    vdw_atom_idx:list, 
                    offset: int,
                    to_default: bool
                    ):
        
        if vdw_atom_idx not in self.atoms_to_be_mutated:
            raise RuntimeError(f'Specified atom {vdw_atom_idx} is not in atom_idx list {self.atoms_to_be_mutated}. Aborting.')

        logger.debug(f"Acting on atoms: {vdw_atom_idx}")
        offset = min([a.idx for a in psf.view[f":{self.tlc.upper()}"].atoms])

        for i in vdw_atom_idx:
            atom = psf[i + offset]
            if to_default:
                psf.mutations_to_default += 1
                atom_type = f'DDX{psf.mutations_to_default}'
                atom.rmin = 1.5
                atom.epsilon = -0.15
            else:
                psf.number_of_dummys += 1
                atom_type = f"DDD{psf.number_of_dummys}"
                self._scale_epsilon(atom, lambda_value)
                self._scale_rmin(atom, lambda_value)
                self._modify_type(atom, psf, atom_type)

    def mutate(self,
               psf: pm.charmm.CharmmPsfFile,
               lambda_value_electrostatic: float = 1.0,
               lambda_value_vdw: float = 1.0,
               vdw_atom_idx: list = [],
               steric_mutation_to_default: bool = False):
        """ Performs the mutation """

        if lambda_value_electrostatic < 0.0 or lambda_value_electrostatic > 1.0:
            raise RuntimeError('Lambda value for LJ needs to be between 0.0 and 1.0.')
        
        if lambda_value_vdw < 0.0 or lambda_value_vdw > 1.0:
            raise RuntimeError('Lambda value for vdw needs to be between 0.0 and 1.0.')
        
        logger.debug(f"LJ scaling factor: {lambda_value_electrostatic}")
        logger.debug(f"VDW scaling factor: {lambda_value_vdw}")

        offset = min([a.idx for a in psf.view[f":{self.tlc.upper()}"].atoms])

        if lambda_value_electrostatic < 1.0:
            self._mutate_charge(psf, lambda_value_electrostatic, offset)

        if lambda_value_vdw < 1.0:
            self._mutate_vdw(psf, lambda_value_vdw, vdw_atom_idx, offset, steric_mutation_to_default)


    def _compensate_charge(self, psf: pm.charmm.CharmmPsfFile, total_charge: int, offset: int):
        """
        _compensate_charge This function compensates the charge changes of a dummy region on the terminal real atom
        that connects the specific dummy group to the real region. 

        Parameters
        ----------
        psf : pm.charmm.CharmmPsfFile
            [description]
        total_charge : int
            [description]
        offset : int
            [description]

        Raises
        ------
        RuntimeError
            [description]
        """
        # get current charge
        new_charge = round(sum([a.charge for a in psf.view[f":{self.tlc.upper()}"].atoms]), 8)
        # get dummy retions
        connected_dummy_regions = self.dummy_region.connected_dummy_regions
        
        # check for each dummy region how much charge has changed and compensate on atom that connects 
        # the real region with specific dummy regions
        for dummy_idx in connected_dummy_regions:
            logger.debug(f'Dummy idx region: {dummy_idx}')
            connecting_real_atom_for_this_dummy_region = self.dummy_region.return_connecting_real_atom(dummy_idx)
            logger.debug(f'Connecting atom: {connecting_real_atom_for_this_dummy_region}')
        
            if connecting_real_atom_for_this_dummy_region is None:
                raise RuntimeError('Something went wrong with the charge compensation. Aborting.')
            charge_to_compenstate_for_region = 0.0
            for atom_idx in dummy_idx:
                charge_to_compenstate_for_region += psf[atom_idx+offset].initial_charge - psf[atom_idx+offset].charge 
            
            logger.info(f'Charge to compensate: {charge_to_compenstate_for_region}')
            psf[connecting_real_atom_for_this_dummy_region+offset].charge += charge_to_compenstate_for_region 
            
            
        # check if rest charge is missing        
        new_charge = round(sum([atom.charge for atom in psf.view[f":{self.tlc.upper()}"].atoms]), 8)

        if not (np.isclose(new_charge, total_charge, rtol=1e-4)):
            raise RuntimeError(f'Charge compensation failed. Introducing non integer total charge: {new_charge}.')

    def _scale_epsilon(self, atom, lambda_value):
        logger.debug(atom)
        logger.debug(atom.initial_epsilon)
        atom.epsilon = atom.initial_epsilon * lambda_value

    def _scale_rmin(self, atom, lambda_value):
        logger.debug(atom)
        logger.debug(atom.initial_rmin)
        atom.rmin = atom.initial_rmin * lambda_value

    def _modify_type(self, atom, psf, new_type: str):

        if (hasattr(atom, 'initial_type')):
            # only change parameters
            pass
        else:
            atom.initial_type = atom.type
            atom.type = new_type
            psf.number_of_dummys += 1
