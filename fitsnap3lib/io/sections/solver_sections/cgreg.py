from fitsnap3lib.io.sections.sections import Section


class Cgreg(Section):

    def __init__(self, name, config, args):
        super().__init__(name, config, args)
        self.allowedkeys = ['alphareg','max_iter','coef_','group_list','tol','updates','preconditioner_type','maxlevel']
        self._check_section()

        self._check_if_used("SOLVER", "solver", "SVD")

        self.alphareg = self.get_value("CGREG", "alphareg", "1.E-5", "float")
        self.max_iter = self.get_value("CGREG", "max_iter", "100000", "int")
        self.updates = self.get_value("CGREG", "updates", "3", "int")
        preconditioner_type = self.get_value("CGREG", "preconditioner_type", "pairwise", "str")
        self.maxlevel = self.get_value("CGREG", "maxlevel", "5", "int")
        assert preconditioner_type in ['pairwise','smoothed','rootnode'], "preconditioner type must be selected from %s" % ', '.join(['pairwise','smoothed','rootnode'])
        self.preconditioner_type = preconditioner_type
        full_group_lst = self.get_value("CGREG", "group_list", "INCR_H-H_0 INCR_H-H_1").split()
        self.full_group_list = full_group_lst
        group_lists = []
        bond_types = [fg.split('_')[1] for fg in full_group_lst]
        bond_types = sorted(list(set(bond_types)))
        for bond_type in bond_types:
            this_bond_lst = []
            for fg in full_group_lst:
                if bond_type in fg:
                    this_bond_lst.append(fg)
            group_lists.append(this_bond_lst)

        self.group_lists = group_lists
        coef_in = self.get_value("CGREG", "coef_", "0.0").split()
        if len(coef_in) == 1:
            self.coef_ = None
        else:
            self.coef_ = [float(k) for k in coef_in]
        self.tol = self.get_value("CGREG", "tol", "1.E-4", "float")
        self.delete()
