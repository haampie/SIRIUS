namespace sirius {

class unit_cell
{
    private:
        
        /// mapping between atom type id and an ordered index in the range [0, N_{types} - 1]
        std::map<int, int> atom_type_index_by_id_;
         
        /// list of atom types
        std::vector<AtomType*> atom_types_;

        /// list of atom classes
        std::vector<AtomSymmetryClass*> atom_symmetry_classes_;
        
        /// list of atoms
        std::vector<Atom*> atoms_;
       
        /// Bravais lattice vectors in row order
        double lattice_vectors_[3][3];
        
        /// inverse Bravais lattice vectors in column order (used to find fractional coordinates by Cartesian coordinates)
        double inverse_lattice_vectors_[3][3];
        
        /// vectors of the reciprocal lattice in row order (inverse Bravais lattice vectors scaled by 2*Pi)
        double reciprocal_lattice_vectors_[3][3];

        /// volume of the unit cell; volume of Brillouin zone is (2Pi)^3 / omega
        double omega_;
       
        /// spglib structure with symmetry information
        SpglibDataset* spg_dataset_;
        
        /// total nuclear charge
        int total_nuclear_charge_;
        
        /// total number of core electrons
        int num_core_electrons_;
        
        /// total number of valence electrons
        int num_valence_electrons_;

        /// total number of electrons
        int num_electrons_;
    
        /*! 
            \brief Get crystal symmetries and equivalent atoms.
            
            Makes a call to spglib providing the basic unit cell information: lattice vectors and atomic types 
            and positions. Gets back symmetry operations and a table of equivalent atoms. The table of equivalent 
            atoms is then used to make a list of atom symmetry classes and related data.
        */
        void get_symmetry()
        {
            Timer t("sirius::unit_cell::get_symmetry");
            
            if (spg_dataset_) 
                error(__FILE__, __LINE__, "spg_dataset is already allocated");
                
            if (atom_symmetry_classes_.size() != 0)
                error(__FILE__, __LINE__, "atom_symmetry_class_by_id_ list is not empty");
            
            if (atoms_.size() == 0)
                error(__FILE__, __LINE__, "atoms_ list is empty");

            double lattice[3][3];

            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++) 
                    lattice[i][j] = lattice_vectors_[j][i];

            mdarray<double,2> positions(NULL, 3, atoms_.size());
            positions.allocate();
            
            std::vector<int> types(atoms_.size());

            for (int i = 0; i < (int)atoms_.size(); i++)
            {
                atoms_[i]->get_position(&positions(0, i));
                types[i] = atoms_[i]->type_id();
            }
            spg_dataset_ = spg_get_dataset(lattice, (double(*)[3])&positions(0, 0), &types[0], atoms_.size(), 1e-5);

            if (spg_dataset_->spacegroup_number == 0)
                error(__FILE__, __LINE__, "spg_get_dataset() returned 0 for the space group");

            if (spg_dataset_->n_atoms != (int)atoms_.size())
                error(__FILE__, __LINE__, "wrong number of atoms");

            AtomSymmetryClass* atom_symmetry_class;
            
            int atom_class_id = -1;

            for (int i = 0; i < (int)atoms_.size(); i++)
            {
                if (atoms_[i]->symmetry_class_id() == -1) // if class id is not assigned to this atom
                {
                    atom_class_id++; // take next id 
                    atom_symmetry_class = new AtomSymmetryClass(atom_class_id, atoms_[i]->type());
                    atom_symmetry_classes_.push_back(atom_symmetry_class);

                    for (int j = 0; j < (int)atoms_.size(); j++) // scan all atoms
                        if (spg_dataset_->equivalent_atoms[j] == spg_dataset_->equivalent_atoms[i]) // assign new class id for all equivalent atoms
                        {
                            atom_symmetry_class->add_atom_id(j);
                            atoms_[j]->set_symmetry_class(atom_symmetry_class);
                        }
                }
            }
        }
        
    public:
    
        unit_cell() : spg_dataset_(NULL)
        {
            assert(sizeof(int4) == 4);
            assert(sizeof(real8) == 8);
        }
       
        void init()
        {
            get_symmetry();
            total_nuclear_charge_ = 0;
            num_core_electrons_ = 0;
            num_valence_electrons_ = 0;

            for (int i = 0; i < num_atoms(); i++)
            {
                total_nuclear_charge_ += atom(i)->type()->zn();
                num_core_electrons_ += atom(i)->type()->num_core_electrons();
                num_valence_electrons_ += atom(i)->type()->num_valence_electrons();
            }
            num_electrons_ = num_core_electrons_ + num_valence_electrons_;
        }

        void clear()
        {
            if (spg_dataset_)
            {
                spg_free_dataset(spg_dataset_);
                spg_dataset_ = NULL;
            }
            
            // delete atom types
            for (int i = 0; i < (int)atom_types_.size(); i++)
                delete atom_types_[i];
            atom_types_.clear();
            atom_type_index_by_id_.clear();

            // delete atom classes
            for (int i = 0; i < (int)atom_symmetry_classes_.size(); i++)
                delete atom_symmetry_classes_[i];
            atom_symmetry_classes_.clear();

            // delete atoms
            for (int i = 0; i < (int)atoms_.size(); i++)
                delete atoms_[i];
            atoms_.clear();
        }

        void print_info()
        {
            printf("lattice vectors\n");
            for (int i = 0; i < 3; i++)
                printf("  a%1i : %18.10f %18.10f %18.10f \n", i + 1, lattice_vectors(i, 0), 
                                                                     lattice_vectors(i, 1), 
                                                                     lattice_vectors(i, 2)); 
            printf("reciprocal lattice vectors\n");
            for (int i = 0; i < 3; i++)
                printf("  b%1i : %18.10f %18.10f %18.10f \n", i + 1, reciprocal_lattice_vectors(i, 0), 
                                                                     reciprocal_lattice_vectors(i, 1), 
                                                                     reciprocal_lattice_vectors(i, 2));
            
            printf("\n"); 
            printf("number of atom types : %i\n", num_atom_types());
            for (int i = 0; i < num_atom_types(); i++)
            {
                int id = atom_type(i)->id();
                printf("type id : %i   symbol : %2s   label : %2s   mt_radius : %10.6f\n", id,
                                                                                           atom_type(i)->symbol().c_str(), 
                                                                                           atom_type(i)->label().c_str(),
                                                                                           atom_type(i)->mt_radius()); 
            }

            printf("number of atoms : %i\n", num_atoms());
            printf("number of symmetry classes : %i\n", num_symmetry_classes());

            printf("\n"); 
            printf("atom id    type id    class id\n");
            printf("------------------------------\n");
            for (int i = 0; i < num_atoms(); i++)
                printf("%6i     %6i      %6i\n", i, atom(i)->type_id(), atom(i)->symmetry_class_id()); 
           
            printf("\n");
            for (int ic = 0; ic < num_symmetry_classes(); ic++)
            {
                printf("class id : %i   atom id : ", ic);
                for (int i = 0; i < atom_symmetry_class(ic)->num_atoms(); i++)
                    printf("%i ", atom_symmetry_class(ic)->atom_id(i));  
                printf("\n");
            }

            printf("\n");
            printf("space group number   : %i\n", spg_dataset_->spacegroup_number);
            printf("international symbol : %s\n", spg_dataset_->international_symbol);
            printf("Hall symbol          : %s\n", spg_dataset_->hall_symbol);
            printf("number of operations : %i\n", spg_dataset_->n_operations);
            
            printf("\n");
            printf("total nuclear charge        : %i\n", total_nuclear_charge_);
            printf("number of core electrons    : %i\n", num_core_electrons_);
            printf("number of valence electrons : %i\n", num_valence_electrons_);
            printf("total number of electrons   : %i\n", num_electrons_);
         }
        
        /*!
            \brief Set lattice vectors.

            Initializes lattice vectors, inverse lattice vector matrix, reciprocal lattice vectors and 
            unit cell volume.

        */
        void set_lattice_vectors(double* a1, 
                                 double* a2, 
                                 double* a3)
        {
            for (int x = 0; x < 3; x++)
            {
                lattice_vectors_[0][x] = a1[x];
                lattice_vectors_[1][x] = a2[x];
                lattice_vectors_[2][x] = a3[x];
            }
            double a[3][3];
            memcpy(&a[0][0], &lattice_vectors_[0][0], 9 * sizeof(double));
            
            double t1 = a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]) + 
                        a[0][1] * (a[1][2] * a[2][0] - a[1][0] * a[2][2]) + 
                        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]);
            
            omega_ = fabs(t1);
            
            if (omega_ < 1e-20)
                error(__FILE__, __LINE__, "lattice vectors are linearly dependent");
            
            t1 = 1.0 / t1;

            double b[3][3];
            b[0][0] = t1 * (a[1][1] * a[2][2] - a[1][2] * a[2][1]);
            b[0][1] = t1 * (a[0][2] * a[2][1] - a[0][1] * a[2][2]);
            b[0][2] = t1 * (a[0][1] * a[1][2] - a[0][2] * a[1][1]);
            b[1][0] = t1 * (a[1][2] * a[2][0] - a[1][0] * a[2][2]);
            b[1][1] = t1 * (a[0][0] * a[2][2] - a[0][2] * a[2][0]);
            b[1][2] = t1 * (a[0][2] * a[1][0] - a[0][0] * a[1][2]);
            b[2][0] = t1 * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
            b[2][1] = t1 * (a[0][1] * a[2][0] - a[0][0] * a[2][1]);
            b[2][2] = t1 * (a[0][0] * a[1][1] - a[0][1] * a[1][0]);

            memcpy(&inverse_lattice_vectors_[0][0], &b[0][0], 9 * sizeof(double));

            for (int l = 0; l < 3; l++)
                for (int x = 0; x < 3; x++)
                    reciprocal_lattice_vectors_[l][x] = twopi * inverse_lattice_vectors_[x][l];
        }
        
        /*!
            \brief Get x coordinate of lattice vector l
         */
        inline double lattice_vectors(int l, int x)
        {
            return lattice_vectors_[l][x];
        }
        
        /*!
            \brief Get x coordinate of reciprocal lattice vector l
         */
        inline double reciprocal_lattice_vectors(int l, int x)
        {
            return reciprocal_lattice_vectors_[l][x];
        }

        /*! 
            \brief Convert coordinates (fractional <-> Cartesian) of direct or reciprocal lattices
        */
        template<coordinates_type cT, lattice_type lT, typename T>
        void get_coordinates(T* a, 
                             double* b)
        {
            b[0] = b[1] = b[2] = 0.0;
            
            if (lT == direct)
            {
                if (cT == fractional)
                {    
                    for (int l = 0; l < 3; l++)
                        for (int x = 0; x < 3; x++)
                            b[l] += a[x] * inverse_lattice_vectors_[x][l];
                }

                if (cT == cartesian)
                {
                    for (int x = 0; x < 3; x++)
                        for (int l = 0; l < 3; l++)
                            b[x] += a[l] * lattice_vectors_[l][x];
                }
            }

            if (lT == reciprocal)
            {
                if (cT == fractional)
                {
                    for (int l = 0; l < 3; l++)
                        for (int x = 0; x < 3; x++)
                            b[l] += lattice_vectors_[l][x] * a[x] / twopi;
                }

                if (cT == cartesian)
                {
                    for (int x = 0; x < 3; x++)
                        for (int l = 0; l < 3; l++)
                            b[x] += a[l] * reciprocal_lattice_vectors_[l][x];
                }
            }
        }

        /*!
            \brief Unit cell volume.
        */
        inline double omega()
        {
            return omega_;
        }

        /*! 
            \brief Add new atom type to the list of atom types.
        */
        void add_atom_type(int atom_type_id, 
                           const std::string& label)
        {
            if (atom_type_index_by_id_.count(atom_type_id) != 0) 
            {   
                std::stringstream s;
                s << "atom type with id " << atom_type_id << " is already in list";
                error(__FILE__, __LINE__, s);
            }
            atom_types_.push_back(new AtomType(atom_type_id, label));
            atom_type_index_by_id_[atom_type_id] = atom_types_.size() - 1;
        }
        
        void add_atom(int atom_type_id, 
                      double* position, 
                      double* vector_field)
        {
            double eps = 1e-10;
            double pos[3];
            
            if (atom_type_index_by_id_.count(atom_type_id) == 0)
            {
                std::stringstream s;
                s << "atom type with id " << atom_type_id << " is not found";
                error(__FILE__, __LINE__, s);
            }
 
            for (int i = 0; i < (int)atoms_.size(); i++)
            {
                atom(i)->get_position(pos);
                if (fabs(pos[0] - position[0]) < eps &&
                    fabs(pos[1] - position[1]) < eps &&
                    fabs(pos[2] - position[2]) < eps)
                {
                    std::stringstream s;
                    s << "atom with the same position is already in list" << std::endl
                      << "  position : " << position[0] << " " << position[1] << " " << position[2];
                    
                    error(__FILE__, __LINE__, s);
                }
            }

            atoms_.push_back(new Atom(atom_type_by_id(atom_type_id), position, vector_field));
        }

        /*!
            \brief Number of atoms in the unit cell.
        */
        inline int num_atoms()
        {
            return atoms_.size();
        }
        
        /*!
            \brief Pointer to atom by atom id.
        */
        inline Atom* atom(int id)
        {
            return atoms_[id];
        }
        
        /*! 
            \brief Number of atom types.
        */
        inline int num_atom_types()
        {
            assert(atom_types_.size() == atom_type_index_by_id_.size());

            return atom_types_.size();
        }

        /*!
            \brief Atom type index by atom type id.
        */
        inline int atom_type_index_by_id(int id)
        {
            return atom_type_index_by_id_[id];
        }
        
        /*!
            \brief Pointer to atom type by type id
        */
        inline AtomType* atom_type_by_id(int id)
        {
            return atom_types_[atom_type_index_by_id(id)];
        }
 
        /*!
            \brief Pointer to atom type by type index (not(!) by atom type id)
        */
        inline AtomType* atom_type(int idx)
        {
            return atom_types_[idx];
        }
       
        /*!
            \brief Number of symmetry classes.
        */
        inline int num_symmetry_classes()
        {
            return atom_symmetry_classes_.size();
        }
       
        /*! 
            \brief Pointer to symmetry class by class id.
        */
        inline AtomSymmetryClass* atom_symmetry_class(int id)
        {
            return atom_symmetry_classes_[id];
        }
        
        inline double num_electrons()
        {
            return num_electrons_;
        }
};

};

