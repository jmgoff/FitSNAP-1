from fitsnap3lib.scrapers.scrape import Scraper, convert
from fitsnap3lib.io.input import Config
from fitsnap3lib.parallel_tools import ParallelTools
from fitsnap3lib.io.output import output
from copy import copy
import os, random, glob, json, datetime ## TODO clean up once done
import numpy as np


config = Config()
pt = ParallelTools()

class Vasp(Scraper):

    def __init__(self, name):
        super().__init__(name)
        pt.single_print("Initializing VASP scraper")
        self.log_data = []
        self.all_data = []
        self.configs = {}
        self.bad_configs = {}
        self.all_config_dicts = []
        self.bc_bool = False
        self.infile = config.args.infile
        self.group_table = config.sections["GROUPS"].group_table
        self.vasp_ignore_incomplete = config.sections["GROUPS"].vasp_ignore_incomplete
        self.vasp_overwrite_jsons = config.sections["GROUPS"].vasp_overwrite_jsons

        ## Before scraping, esnure that user has correct input
        ## TODO: Logan recently fixed this, check before putting in again
        # self.check_train_test_sizes()


    def scrape_groups(self):
        ### Locate all OUTCARs in datapath
        ## TODO rework pathing/glob with os.path.join() to make system agnostic
        glob_asterisks = '/**/*'
        outcars_base = config.sections['PATH'].datapath + glob_asterisks
        ## TODO make this search user-specify-able
        all_outcars = [f for f in glob.glob(outcars_base,recursive=True) if f.endswith('OUTCAR')]

        ## Grab test|train split
        self.group_dict = {k: config.sections['GROUPS'].group_types[i] for i, k in enumerate(config.sections['GROUPS'].group_sections)}
        for group in self.group_table:
            training_size = None
            if 'size' in self.group_table[group]:
                training_size = self.group_table[group]['size']
                self.bc_bool = True
            if 'training_size' in self.group_table[group]:
                if training_size is not None:
                    raise ValueError('Do not set both size and training size')
                training_size = self.group_table[group]['training_size']
                #size_type = group_dict['training_size']
            if 'testing_size' in self.group_table[group]:
                testing_size = self.group_table[group]['testing_size']
                testing_size_type = self.group_dict['testing_size']
            else:
                testing_size = 0
            if training_size is None:
                raise ValueError('Please set training size for {}'.format(group))
            
            ## Grab OUTCARS for this training group
            ## TODO emulate XYZ pop() to ensure items are processed once only
            group_outcars = [f for f in all_outcars if group in f]

            file_base = os.path.join(config.sections['PATH'].datapath, group)
            self.files[file_base] = group_outcars
            self.configs[group] = []  ##TODO ? need this? copied from XYZ

            try:
                for outcar in self.files[file_base]:
                    ## Open file
                    with open(outcar, 'r') as fp:
                        lines = fp.readlines()
                    nlines = len(lines)

                    ## Use ion loop text to partition ionic steps
                    ion_loop_text = 'aborting loop because EDIFF is reached'
                    start_idx_loops = [i for i, line in enumerate(lines) if ion_loop_text in line]
                    end_idx_loops = [i for i in start_idx_loops[1:]] + [nlines]

                    ## Grab potcar and element info
                    header_lines = lines[:start_idx_loops[0]]
                    potcar_list, potcar_elements, ions_per_type = self.parse_outcar_header(header_lines)

                    ## Each config in a single OUTCAR is assigned the same
                    ## parent data (i.e. filename, potcar and ion data)
                    ## but separated for each iteration (idx loops on 'lines')
                    ## Tuple data: outcar file name str, config number int, starting line number (for debug)  int, 
                    ## potcar list, potcar elements list, number ions per element list, configuration lines list 
                    unique_configs = [(outcar, i, start_idx_loops[i], potcar_list, potcar_elements, ions_per_type,
                                        lines[start_idx_loops[i]:end_idx_loops[i]])
                                        for i in range(0, len(start_idx_loops))]
                    for uc in unique_configs:
                        config_dict = self.generate_outcar_dict(group, uc)
                        if config_dict != -1:
                            self.configs[group].append(config_dict)
            except IndexError:
                ## TODO what does this IndexError take care of? (inherited from Charlie's code)
                self.configs[file_base].pop(-1)

            if config.sections["GROUPS"].random_sampling:
                random.shuffle(self.configs[group], pt.get_seed)
            nconfigs = len(self.configs[group])

            ## Assign configurations to train/test groups
            ## check_train_test_sizes() confirms that training_size > 0 and
            ## that training_size + testing_size = 1.0
            ## TODO make sure this doesn't conflict with Logan's fix
            if training_size == 1:
                training_configs = nconfigs
                testing_configs = 0
            else:
                training_configs = max(1, int(round(training_size * nconfigs)))
                if training_configs == nconfigs:
                    ## If training_size is not exactly 1.0, add at least 1 testing config
                    training_configs -= 1
                    testing_configs = 1
                else:
                    testing_configs = nconfigs - training_configs

            if nconfigs - testing_configs - training_configs < 0:
                raise ValueError("training configs: {} + testing configs: {} is greater than files in folder: {}".format(
                    training_configs, testing_configs, nconfigs))

            output.screen(f"{group}: Detected {nconfigs}, fitting on {training_configs}, testing on {testing_configs}")

            ## Populate tests dictionary
            if self.tests is None:
                self.tests = {}
            self.tests[group] = []

            ## Removed next two lines since we gracefully crash if train/test not OK
            # for i in range(nconfigs - training_configs - testing_configs):
            #     self.configs[group].pop()
            for i in range(testing_configs):
                self.tests[group].append(self.configs[group].pop())

            ## TODO propagate change of variable from "_size" to "_configs" or something similar
            self.group_table[group]['training_size'] = training_configs
            self.group_table[group]['testing_size'] = testing_configs


    def scrape_configs(self):
        """Generate and send (mutable) data to send to fitsnap"""
        ## TODO clean up as we have already run many of the asertions by this point
        ## TODO maybe just read JSONs for now...?
        """ Copied almost exactly from json_scraper.py"""
        self.conversions = copy(self.default_conversions)
        for i, data0 in enumerate(self.configs):
            data = data0[0]
            
            assert len(data) == 1, "More than one object (dataset) is in this file"

            self.data = data['Dataset']

            assert len(self.data['Data']) == 1, "More than one configuration in this dataset"

            assert all(k not in self.data for k in self.data["Data"][0].keys()), \
                "Duplicate keys in dataset and data"

            self.data.update(self.data.pop('Data')[0])  # Move self.data up one level

            for key in self.data:
                if "Style" in key:
                    if key.replace("Style", "") in self.conversions:
                        temp = config.sections["SCRAPER"].properties[key.replace("Style", "")]
                        temp[1] = self.data[key]
                        self.conversions[key.replace("Style", "")] = convert(temp)

            for key in config.sections["SCRAPER"].properties:
                if key in self.data:
                    self.data[key] = np.asarray(self.data[key])

            natoms = np.shape(self.data["Positions"])[0]
            pt.shared_arrays["number_of_atoms"].sliced_array[i] = natoms
            self.data["QMLattice"] = (self.data["Lattice"] * self.conversions["Lattice"]).T
            del self.data["Lattice"]  # We will populate this with the lammps-normalized lattice.
            if "Label" in self.data:
                del self.data["Label"]  # This comment line is not that useful to keep around.

            if not isinstance(self.data["Energy"], float):
                self.data["Energy"] = float(self.data["Energy"])

            # Currently, ESHIFT should be in units of your training data (note there is no conversion)
            if hasattr(config.sections["ESHIFT"], 'eshift'):
                for atom in self.data["AtomTypes"]:
                    self.data["Energy"] += config.sections["ESHIFT"].eshift[atom]

            self.data["test_bool"] = self.test_bool[i]

            self.data["Energy"] *= self.conversions["Energy"]

            self._rotate_coords()
            self._translate_coords()

            self._weighting(natoms)

            self.all_data.append(self.data)

        return self.all_data

    def generate_outcar_dict(self, group, outcar_config):
        config_dict = {}
        is_bad_config = False
        outcar_filename, config_num, start_idx, potcar_list, potcar_elements, ions_per_type, lines = outcar_config
        # # group = outcar_config[1]
        # print(lines)
        # print(group, outcar_filename, config_num, start_idx, potcar_list, potcar_elements, ions_per_type)
        # exit()
        config_data = self.parse_outcar_config(lines, potcar_list, potcar_elements, ions_per_type)
        if type(config_data) == tuple:
            crash_type, crash_line = config_data
            is_bad_config = True
            ## TODO sort out 'bad' OUTCARs earlier
            if not self.vasp_ignore_incomplete:
                raise Exception('!!ERROR: OUTCAR step incomplete!!' \
                    '\n!!Not all atom coordinates/forces were written to a configuration' 
                    '\n!!Please check the OUTCAR for incomplete steps and adjust, '
                    '\n!!or toggle variable "vasp_ignore_incomplete" to True'
                    '\n!!(not recommended as you may miss future incomplete steps)' 
                    f'\n!!\tOUTCAR location: {outcar_filename}' 
                    f'\n!!\tConfiguration number: {config_num}' 
                    f'\n!!\tLine number of error: {start_idx}' 
                    f'\n!!\tExpected {crash_type}, {crash_line} '
                    '\n')
            else:
                output.screen('!!WARNING: OUTCAR step incomplete!!'
                    '\n!!Not all atom coordinates/coordinates were written to a configuration'
                    '\n!!Variable "vasp_ignore_incomplete" is toggled to True'
                    '\n!!Note that this may result in missing training set data (e.g., missing final converged structures)'
                    f'\n!!\tOUTCAR location: {outcar_filename}' 
                    f'\n!!\tConfiguration number: {config_num}'
                    f'\n!!\tLine number of warning: {start_idx}'
                    f'\n!!\tExpected {crash_type}, {crash_line} '
                    '\n')

        config_header = {}
        config_header['Group'] = group
        config_header['File'] = outcar_filename
        config_header['EnergyStyle'] = "electronvolt"
        config_header['StressStyle'] = "kB"
        config_header['AtomTypeStyle'] = "chemicalsymbol"
        config_header['PositionsStyle'] = "angstrom"
        config_header['ForcesStyle'] = "electronvoltperangstrom"
        config_header['LatticeStyle'] = "angstrom"
        config_header['Data'] = [config_data]

        config_dict['Dataset'] = config_header

        if not is_bad_config:
            # outcar_name, outcar_data, json_path, json_filestem, file_num
            json_path = f'JSON/{group}'
            if not os.path.exists(json_path):
                os.makedirs(json_path)
            file_num = config_num + 1
            json_filestem = outcar_filename.replace('/','_').replace('_OUTCAR','') #.replace(f'_{group}','')
            json_filename = f"{json_path}/{json_filestem}{file_num}.json"
            if not os.path.exists(json_filename) or self.vasp_overwrite_jsons:
                self.write_json(json_filename, outcar_filename, config_dict)
            return config_dict
        else:
            return -1

    def parse_outcar_config(self,lines,potcar_list,potcar_elements,ions_per_type):
        ## TODO clean up syntax to match FitSNAP3
        ## TODO clean up variable names to match input, increase clarity
        ## LIST SECTION_MARKERS AND RELATED FUNCTIONS ARE HARD-CODED!!
        ## DO NOT CHANGE UNLESS YOU KNOW WHAT YOU'RE DOING!!

        section_markers = [
            'FORCE on cell',
            'direct lattice vectors',
            'TOTAL-FORCE (eV/Angst)',
            'FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)',
        ]

        section_names = [
            'stresses',
            'lattice',
            'coords & forces',
            'energie',
        ]

        idx_stress_vects = 0 # 4
        idx_lattice_vects = 1 # 5
        idx_force_vects = 2 # 6
        idx_energie = 3 # 7

        ## Index lines of file containing JSON data
        section_idxs = [None,None,None,None]
        atom_coords, atom_forces, stress_component, all_lattice, total_energie  = None, None, None, None, None

        list_atom_types = []
        for i, elem in enumerate(potcar_elements):
            num_repeats = ions_per_type[i]
            elem_list = [elem.strip()]*num_repeats
            list_atom_types.extend(elem_list)
        natoms = sum(ions_per_type)

        ## Search entire file to create indices for each section
        for i, line in enumerate(lines):
            ## TODO refactor - probably a smarter/faster way to do the "line_test" part...
            line_test = [True if sm in line else False for sm in section_markers]
            if any(line_test):
                test_idx = [n for n, b in enumerate(line_test) if b][0]
                section_idxs[test_idx] = i
        
        ## If this config has any sections missing, it is incomplete, return crash
        missing_sections = [True if i == None else False for i in section_idxs]
        if any(missing_sections):
            crash_type = '4 sections'
            missing_sections_str = 'missing sections: '
            missing_sections_str += ', '.join([section_names[i] for i, b in enumerate(missing_sections) if b])
            del lines
            return (crash_type, missing_sections_str)

        ## Create data dict for this config, with global information already included
        data = {}
        data['AtomTypes'] = list_atom_types  ## orig in poscar, done
        data['NumAtoms'] = natoms  ## orig in poscar, done

        ## Lattice vectors in real space
        ## Note: index to initial lattice vector output (POSCAR) in OUTCAR has already been removed.
        ## Actual vector starts one line after that, and has 3 lines
        lidx_last_lattice0 = section_idxs[idx_lattice_vects] + 1
        lidx_last_lattice1 = lidx_last_lattice0 + 3
        lines_last_lattice = lines[lidx_last_lattice0:lidx_last_lattice1]
        all_lattice = self.get_direct_lattice(lines_last_lattice)

        ## Stresses
        lidx_stresses = section_idxs[idx_stress_vects] + 14
        line_stresses = lines[lidx_stresses]
        stress_component = self.get_stresses(line_stresses)

        ## Atom coordinates and forces
        lidx_forces0 = section_idxs[idx_force_vects] + 2
        lidx_forces1 = lidx_forces0 + natoms
        lines_forces = lines[lidx_forces0:lidx_forces1]
        atom_coords, atom_forces = self.get_forces(lines_forces)
        if type(atom_coords) == str:
            crash_type = 'atom coords, atom forces'
            crash_atom_coord_line = 'found bad line: ' + atom_coords
            del lines
            return (crash_type, crash_atom_coord_line)

        ## Energie :-)
        ## We are getting the value without entropy
        lidx_energie = section_idxs[idx_energie] + 4
        line_energie = lines[lidx_energie]
        total_energie = self.get_energie(line_energie)

        # Here is where all the data is put together since the energy value is the last
        # one listed in each configuration.  After this, all these values will be overwritten
        # once the next configuration appears in the sequence when parsing
        data['Positions'] = atom_coords
        data['Forces'] = atom_forces
        data['Stress'] = stress_component
        data['Lattice'] = all_lattice
        data['Energy'] = total_energie
        data["computation_code"] = "VASP"
        data["pseudopotential_information"] = potcar_list

        ## Clean up (othewrise we get memory errors)
        del lines

        return data

    def parse_outcar_header(self, header):
        ## These searches replace the POSCAR and POTCAR, and can also check IBRION for AIMD runs (commented out now)
        lines_potcar, lines_vrhfin, lines_ions_per_type = [], [],[]
        potcar_list, potcar_elements, ions_per_type = [], [], []
        # line_ibrion, is_aimd = "", False

        for line in header:
            if "VRHFIN" in line:
                lines_vrhfin.append(line)
            elif "ions per type" in line:
                lines_ions_per_type.append(line)
            elif "POTCAR" in line:
                lines_potcar.append(line)
                # Look for the ordering of the atom types - grabbing POTCAR filenames first, then atom labels separately because VASP has terribly inconsistent formatting
                if line.split()[1:] not in potcar_list:  # VASP will have these lines in the OUTCAR twice, and we don't need to append them the second time
                    potcar_list.append(line.split()[1:])  # each line will look something like ['PAW_PBE', 'Zr_sv_GW', '05Dec2013']

        ## TODO add check that warns user if POSCAR elements and POTCAR order are not the same (if possible)

        for line in lines_vrhfin:
            str0 = line.strip().replace("VRHFIN =", "")
            str1 = str0[:str0.find(":")]
            potcar_elements.append(str1)

        for line in lines_ions_per_type:
            str0 = line.replace("ions per type = ","").strip()
            ions_per_type = [int(s) for s in str0.split()]

        return potcar_list, potcar_elements, ions_per_type

    def get_vrhfin(self, lines):
        ## Scrapes vrhfin lines to get elements
        ## These lines appear only once per element in OUTCARs
        ## Format: VRHFIN =W: 5p6s5d
        elem_list = []
        for line in lines:
            str0 = line.strip().replace("VRHFIN =", "")
            str1 = str0[:str0.find(":")]
            elem_list.append(str1)
        return elem_list

    def get_ions_per_type(self, lines):
        ions_per_type = []
        for line in lines:
            str0 = line.replace("ions per type = ","").strip()
            ions_per_type = [int(s) for s in str0.split()]
        return ions_per_type

    def get_ibrion(self, line):
        ## There should be only one of these lines (from INCAR print)
        ## IBRION value should always be first number < 10 to appear after "="
        line1 = line.split()
        idx_equals = line1.index("=")
        probably_ibrion = line1[idx_equals+1]
        if probably_ibrion.isdigit():
            if probably_ibrion == "0":
                is_aimd = True ## https://www.vasp.at/wiki/index.php/IBRION
            else:
                is_aimd = False
        else:
            output.screen("!!WARNING: incomplete coding with scrape_ibrion, assuming not AIMD for now.")
            is_aimd = False
        return is_aimd

    def get_direct_lattice(self, lines):
        lattice_coords = []
        for i in range(0, 3):
            lattice_coords.append([float(v) for v in lines[i].split()[:3]])
        return lattice_coords

    def get_stresses(self, line):
        ## TODO check that we can assume symmetric stress tensors
        ## TODO where do we set the cell type (Bravais)
        columns = line.split()
        stress_xx, stress_yy, stress_zz = [float(c) for c in columns[2:5]]
        stress_xy, stress_yz, stress_zx = [float(c) for c in columns[5:8]]
        stresses = [[stress_xx, stress_xy, stress_zx],
                    [stress_xy, stress_yy, stress_yz],
                    [stress_zx, stress_yz, stress_zz]]
        return stresses

    def get_forces(self, lines):
        coords, forces = [], []
        try:
            [float(v) for v in lines[-1].split()[:6]]
        except:
            print('OUTCAR config did not run to completion! Discarding configuration')
            return lines[-1],-1 ## returning faulty string for error message

        for line in lines:
            x, y, z, fx, fy, fz = [float(v) for v in line.split()[:6]]
            coords.append([x, y, z])
            forces.append([fx, fy, fz])
        return coords, forces

    def get_energie(self, line):
        str0 = line[:line.rfind("energy(sigma->")].strip()
        str1 = "".join([c for c in str0 if c.isdigit() or c == "-" or c == "."])
        energie = float(str1)
        return energie
    

    ## TODO create naming scheme
    def write_json(self, json_filename, outcar_filename, config_dict):
        ## Credit for next section goes to Mary Alice Cusentino's VASP2JSON script!
        dt = datetime.datetime.now().strftime('%B %d %Y %I:%M%p')
        comment_line = f'# Generated on {dt} from: {os.getcwd()}/{outcar_filename}'

        ## TODO move comment line from front to inside of JSON object
        ## TODO make sure all JSON reading by FitSNAP is compatible with both formats
        # Comment line at top breaks the JSON format and readers complain
        # with open(json_filename, "w") as f:
        #     f.write(comment_line + "\n")

        ## Write actual JSON object
        # with open(json_filename, "a+") as f: ## with comment line
        with open(json_filename, "w") as f:
            json.dump(config_dict, f, indent=2, sort_keys=True)
        return


## ------------------------ Ideas/plans for future versions ------------------------ ##
    def scrape_incar(self):
        ## Might be able to add this info to FitSNAP JSONs easily, will need to check compatibility
        pass


    def only_read_JSONs_if_OUTCAR_already_converted(self):
        ## Many OUTCARs (esp. for AIMD) are HUGE and take a long time to parse. Design a user-friendly and elegant way to confirm that OUTCAR has already been parsed and only read available JSON data (see above)
        pass

    def check_OUTCAR_filesizes(self):
        ## Many OUTCARs (esp. for AIMD) are HUGE and take a long time to parse. In these cases, strongly recommend to user to toggle vasp2json on, and then use JSONs only, or have an elegant way to only read available JSON data (see above)
        pass

    def generate_lammps_test(self):
        ## Maybe create option where we can take scraped OUTCARs and make LAMMPS-compatible *dat files right away
        pass

    def vasp_namer(self):
        ## Tryiing to think of a clever naming scheme so that users can trace back where they got the file
        return

    # def generate_FitSNAP_JSONs(self):
    #     ## OLD VERSION: keep for naming/JSON-checking scheme
    #     new_converted, bad_outcar_not_converted, already_converted = 0, 0, 0
    #     json_path = config.sections['PATH'].datapath
    #     if not os.path.exists(json_path):
    #         os.mkdir(json_path)
    #     for group, outcars in self.outcars_per_group.items():
    #         ## Check group and group path, create if it doesn't exist
    #         json_group_path = json_path + "/" + group
    #         if not os.path.exists(json_group_path):
    #             os.mkdir(json_group_path)

    #         ## Begin OUTCAR processing
    #         for i, outcar in enumerate(outcars):
    #             pt.single_print(f"Reading: {outcar}")
    #             ## Get OUTCAR directory name for labeling (e.g. default naming scheme, sorting, etc.)
    #             outcar_path_stem = outcar.replace("/OUTCAR", "")[outcar.replace("/OUTCAR", "").rfind("/") + 1:]

    #             ## Create stem for JSON file naming
    #             if self.only_label:
    #                 json_file_stem = f"{json_group_path}/{self.json_label}{i}"
    #             else:
    #                 json_file_stem = f"{json_group_path}/{self.json_label}{i}_{outcar_path_stem}"
    #             pt.single_print(f"\tNew JSON group path and file name(s): {json_file_stem}_*.json ")

    #             ## Find existing JSON files
    #             json_files = glob(json_file_stem + "*.json")

    #             ## Begin converting OUTCARs to FitSNAP JSON format
    #             ## Credit for next sections goes to Mary Alice Cusentino's VASP2JSON script!
    #             if not json_files or self.overwrite:
    #                 ## Reading/scraping of outcar
    #                 data_outcar_configs, num_configs = self.scrape_outcar(outcar)

    #                 ## Check that all expected data in configs from OUTCAR is present
    #                 for n, data in enumerate(data_outcar_configs):
    #                     m = n + 1
    #                     if any([True if val is None else False for val in data.values()]):
    #                         pt.single_print(
    #                             f"!!WARNING: OUTCAR file is missing data: {outcar} \n"
    #                             f"!!WARNING: Continuing without writing JSON...\n")
    #                         self.bad_configs[group] = self.bad_configs[group] + [outcar]
    #                         bad_outcar_not_converted += 1
    #                         status = "could_not_convert"
    #                     else:
    #                         self.write_json(outcar, data, json_file_stem, m)
    #                         new_converted += 1
    #                         status = "new_converted"
    #             else:
    #                 already_converted += 1
    #                 status = "already_converted"
    #             log_info = [group, outcar, outcar_path_stem, json_file_stem, num_configs, status]
    #             self.log_data.append(log_info)

    #     pt.single_print(f"Completed writing JSON files. Summary: \n"
    #                     f"\t\t{new_converted} new JSON files created \n"
    #                     f"\t\t{already_converted} OUTCARs already converted \n"
    #                     f"\t\t{bad_outcar_not_converted} OUTCARs could not be converted \n"
    #                     f"\t\tSee {self.log_file} for more details.\n")

