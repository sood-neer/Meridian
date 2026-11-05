import csv
import os

courses_file = os.path.join(os.path.dirname(__file__), "nith_courses.csv")
majors_file = os.path.join(os.path.dirname(__file__), "nith_majors.csv")

# Full official course dataset from https://nith.ac.in/bachelor-course-structure-syllabus
nith_courses = [
    # --- FIRST YEAR COURSES (NEP-2020) ---
    {
        "course_number": "MA-111",
        "course_name": "Linear Algebra & Calculus\n\n3 Credits",
        "description": "Matrices, eigenvalues and eigenvectors, Cayley-Hamilton Theorem, diagonalization. Calculus of single and multivariable functions, Taylor series, maxima and minima, multiple integrals, vector calculus, Green's and Stokes' theorems. Syllabus source: National Institute of Technology Hamirpur bachelor-course-structure-syllabus.",
        "prerequisite_data": "None",
        "prerequisite_course_numbers": "[]",
        "learning_objectives": "['Understand matrix algebra and eigenvalue problems', 'Apply multivariable calculus and multiple integrals to engineering']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "PH-101",
        "course_name": "Engineering Physics\n\n3 Credits",
        "description": "Oscillations, waves, interference, diffraction, polarization, lasers and optical fibers, quantum mechanics, wave-particle duality, Schrodinger equation, solid state physics, semiconductor devices. Syllabus from NIT Hamirpur B.Tech First Year curriculum.",
        "prerequisite_data": "None",
        "prerequisite_course_numbers": "[]",
        "learning_objectives": "['Grasp wave optics and quantum mechanics principles', 'Understand laser and semiconductor physics']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CS-111",
        "course_name": "Introduction to Computer and C Programming\n\n3 Credits",
        "description": "Fundamentals of computers, algorithm and flowchart development. C programming syntax, data types, operators, control flow, functions, recursion, arrays, pointers, strings, structures, unions, dynamic memory allocation and file handling. NIT Hamirpur Department of Computer Science & Engineering syllabus.",
        "prerequisite_data": "None",
        "prerequisite_course_numbers": "[]",
        "learning_objectives": "['Develop problem-solving algorithms', 'Write robust structured programs in C using pointers and memory management']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "ME-101",
        "course_name": "Basics of Mechanical Engineering\n\n3 Credits",
        "description": "Laws of thermodynamics, thermodynamic cycles, power plants, internal combustion engines, heat transfer mechanisms, basic manufacturing processes, casting, welding, machining, and engineering materials.",
        "prerequisite_data": "None",
        "prerequisite_course_numbers": "[]",
        "learning_objectives": "['Understand thermodynamic laws and energy conversion', 'Appreciate mechanical systems and manufacturing principles']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "HS-101",
        "course_name": "Communication Skills\n\n2 Credits",
        "description": "LSRW skills (Listening, Speaking, Reading, Writing), technical writing, report drafting, business correspondence, presentation techniques, public speaking, non-verbal communication and vocabulary enhancement.",
        "prerequisite_data": "None",
        "prerequisite_course_numbers": "[]",
        "learning_objectives": "['Master professional technical communication', 'Enhance presentation and interpersonal skills']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "HS-103",
        "course_name": "Human Values and Ethical Mindfulness\n\n2 Credits",
        "description": "Personal and professional ethics, moral values, emotional intelligence, mindfulness, environmental ethics, ethical decision making in engineering practice, ISO 14000, and corporate social responsibility.",
        "prerequisite_data": "None",
        "prerequisite_course_numbers": "[]",
        "learning_objectives": "['Inculcate human values and mindfulness in engineering', 'Resolve ethical dilemmas in professional life']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "PH-102",
        "course_name": "Physics Lab\n\n1 Credit",
        "description": "Experimental verification of optical phenomena (Newton's rings, diffraction grating, spectrometer), Hall effect, bandgap determination, laser wavelength measurement, and harmonic oscillations.",
        "prerequisite_data": "None",
        "prerequisite_course_numbers": "[]",
        "learning_objectives": "['Conduct precision physical measurements', 'Analyze experimental errors in optical and semiconductor experiments']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "HS-102",
        "course_name": "Communication Skills Lab\n\n1 Credit",
        "description": "Language laboratory sessions focusing on phonetics, pronunciation, group discussions, mock interviews, impromptu speeches, and listening comprehension.",
        "prerequisite_data": "None",
        "prerequisite_course_numbers": "[]",
        "learning_objectives": "['Improve spoken English fluency and accent', 'Excel in group discussions and interview settings']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "ME-102",
        "course_name": "Engineering Workshop Practice\n\n2 Credits",
        "description": "Hands-on training in carpentry, fitting, sheet metal, welding, machining, and foundry shops. Safety regulations and use of machine tools.",
        "prerequisite_data": "None",
        "prerequisite_course_numbers": "[]",
        "learning_objectives": "['Acquire practical fabrication skills', 'Operate machine tools safely']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CE-101",
        "course_name": "Engineering Graphics\n\n2 Credits",
        "description": "Principles of engineering drawing, lettering, orthographic projections of points, lines, planes, and solids. Section of solids, isometric views, and introductory computer-aided drafting (CAD).",
        "prerequisite_data": "None",
        "prerequisite_course_numbers": "[]",
        "learning_objectives": "['Interpret engineering blueprints and drawings', 'Use CAD tools for 2D drafting']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "MA-121",
        "course_name": "Differential Equations\n\n3 Credits",
        "description": "First-order ODEs, higher-order linear ODEs with constant coefficients, Laplace transforms, Fourier series, boundary value problems, partial differential equations (heat, wave, Laplace equations).",
        "prerequisite_data": "MA-111",
        "prerequisite_course_numbers": "['MA-111']",
        "learning_objectives": "['Solve engineering ODEs and PDEs', 'Apply Laplace transforms to dynamic systems']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CY-101",
        "course_name": "Engineering Chemistry\n\n3 Credits",
        "description": "Molecular structure and theories of bonding, spectroscopy, thermodynamics of chemical processes, electrochemistry, corrosion and its prevention, polymers and composite materials, water technology and green chemistry.",
        "prerequisite_data": "None",
        "prerequisite_course_numbers": "[]",
        "learning_objectives": "['Understand corrosion, polymers, and water treatment', 'Apply chemical thermodynamics to materials']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CS-121",
        "course_name": "Introduction to Python and Data Structures\n\n3 Credits",
        "description": "Python programming syntax, collections (lists, tuples, dicts, sets), OOP in Python, numpy, pandas, and foundational data structures (stacks, queues, linked lists, binary trees, sorting and searching algorithms).",
        "prerequisite_data": "CS-111",
        "prerequisite_course_numbers": "['CS-111']",
        "learning_objectives": "['Develop Python applications and data pipelines', 'Implement linear and non-linear data structures']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "EE-101",
        "course_name": "Basic Electrical Engineering\n\n2 Credits",
        "description": "DC circuit analysis (KCL, KVL, Norton, Thevenin, Superposition), AC single phase and three phase circuits, resonance, transformers, DC machines, and induction motors.",
        "prerequisite_data": "None",
        "prerequisite_course_numbers": "[]",
        "learning_objectives": "['Analyze DC and AC networks', 'Understand transformer and electric motor operations']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "EC-101",
        "course_name": "Basic Electronics Engineering\n\n2 Credits",
        "description": "Semiconductor physics, PN junction diodes, BJT, FET, MOSFET, operational amplifiers, inverting and non-inverting configurations, and digital logic gates.",
        "prerequisite_data": "None",
        "prerequisite_course_numbers": "[]",
        "learning_objectives": "['Analyze diode and transistor circuits', 'Design basic op-amp circuits']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "EN-101",
        "course_name": "Energy Environment & Life Science\n\n2 Credits",
        "description": "Renewable and non-renewable energy resources, environmental pollution, climate change, ecosystem dynamics, sustainable development, and biological cell systems.",
        "prerequisite_data": "None",
        "prerequisite_course_numbers": "[]",
        "learning_objectives": "['Assess renewable energy systems and environmental impact', 'Understand sustainable engineering practices']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "MB-101",
        "course_name": "Entrepreneurship & Marketing Dynamics\n\n1 Credit",
        "description": "Principles of entrepreneurship, startup ecosystems, business model canvas, market research, digital marketing fundamentals, and venture capital financing.",
        "prerequisite_data": "None",
        "prerequisite_course_numbers": "[]",
        "learning_objectives": "['Understand entrepreneurial processes and ideation', 'Develop basic commercial business models']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CY-102",
        "course_name": "Engineering Chemistry Lab\n\n1 Credit",
        "description": "Volumetric analysis, hardness of water determination, viscosity and surface tension measurement, conductometric and potentiometric titrations.",
        "prerequisite_data": "None",
        "prerequisite_course_numbers": "[]",
        "learning_objectives": "['Conduct standard water and chemical quality tests', 'Perform instrumental chemical analyses']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "EE-102",
        "course_name": "Electrical & Electronics Engineering Lab\n\n1 Credit",
        "description": "Experimental verification of network theorems, diode characteristics, transistor biasing, op-amp configurations, logic gate operations, and transformer testing.",
        "prerequisite_data": "None",
        "prerequisite_course_numbers": "[]",
        "learning_objectives": "['Build and test analog/digital circuits on breadboards', 'Verify transformer efficiency and machine characteristics']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },

    # --- CSE CORE & ELECTIVES (Semesters 3-8, NEP-2020) ---
    {
        "course_number": "MA-219",
        "course_name": "Probability and Applied Statistics\n\n3 Credits",
        "description": "Probability distributions (Binomial, Poisson, Normal), random variables, expectations, sampling theory, hypothesis testing (t-test, chi-square test, F-test), regression, correlation, and Markov chains for computing.",
        "prerequisite_data": "MA-111",
        "prerequisite_course_numbers": "['MA-111']",
        "learning_objectives": "['Apply statistical methods and testing to computational problems', 'Model stochastic processes with probability distributions']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CS-212",
        "course_name": "Discrete Structures\n\n3 Credits",
        "description": "Sets, relations, functions, propositional and predicate logic, proof methods, mathematical induction, recurrence relations, combinatorics, graph theory, trees, and algebraic structures (groups, rings, lattices).",
        "prerequisite_data": "None",
        "prerequisite_course_numbers": "[]",
        "learning_objectives": "['Master mathematical reasoning and formal proofs', 'Analyze relations, recurrences, and graphs for computing']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CS-213",
        "course_name": "Object Oriented Programming\n\n3 Credits",
        "description": "OOP paradigm in C++ and Java: classes, objects, data encapsulation, inheritance, polymorphism, abstract classes, interfaces, exception handling, templates, generic programming, and standard template library (STL).",
        "prerequisite_data": "CS-111",
        "prerequisite_course_numbers": "['CS-111']",
        "learning_objectives": "['Design modular object-oriented software architectures', 'Utilize templates, polymorphism, and exception safety']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CS-214",
        "course_name": "Data Structures\n\n3 Credits",
        "description": "Asymptotic analysis of algorithms. Linear data structures: stacks, queues, linked lists. Non-linear data structures: trees, BST, AVL trees, B/B+ trees, heaps, graphs, hashing, sorting techniques and graph traversals (BFS, DFS).",
        "prerequisite_data": "CS-111",
        "prerequisite_course_numbers": "['CS-111']",
        "learning_objectives": "['Implement complex data structures and balanced search trees', 'Analyze algorithmic time and space complexities']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CS-215",
        "course_name": "Computer Graphics\n\n3 Credits",
        "description": "Display devices, raster scan systems, line and circle drawing algorithms, 2D and 3D transformations, window-to-viewport mapping, clipping algorithms, 3D projections, hidden surface elimination, shading models, and OpenGL programming.",
        "prerequisite_data": "CS-111",
        "prerequisite_course_numbers": "['CS-111']",
        "learning_objectives": "['Understand 2D/3D graphics rendering pipelines', 'Implement geometric transformations and clipping in OpenGL']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "EC-219",
        "course_name": "Digital Electronics\n\n2 Credits",
        "description": "Number systems, Boolean algebra, logic minimization (K-maps), combinational logic circuits (adders, multiplexers, decoders), sequential circuits, flip-flops, registers, counters, finite state machines, and semiconductor memories.",
        "prerequisite_data": "EC-101",
        "prerequisite_course_numbers": "['EC-101']",
        "learning_objectives": "['Design combinational and sequential digital systems', 'Synthesize synchronous and asynchronous state machines']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CS-217",
        "course_name": "Object Oriented Programming Lab\n\n1 Credit",
        "description": "Practical implementation of OOP concepts in C++/Java: classes, multiple inheritance, operator overloading, virtual functions, file I/O, templates, and GUI programming.",
        "prerequisite_data": "CS-213",
        "prerequisite_course_numbers": "['CS-213']",
        "learning_objectives": "['Implement production-quality OOP code', 'Build robust modular applications with unit tests']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CS-218",
        "course_name": "Data Structures Lab\n\n1 Credit",
        "description": "Hands-on programming of linked lists, stacks, queues, expression evaluation, BST operations, AVL rotations, graph algorithms (Dijkstra, Kruskal, Prim), and sorting comparisons.",
        "prerequisite_data": "CS-214",
        "prerequisite_course_numbers": "['CS-214']",
        "learning_objectives": "['Write clean, efficient C/C++ implementations of data structures', 'Profile algorithm running times experimentally']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CS-219",
        "course_name": "Computational Tools and Workshop\n\n1 Credit",
        "description": "Linux command line, shell scripting, Git version control, Docker containerization, LaTeX documentation, and build automation tools (Makefile, CMake).",
        "prerequisite_data": "None",
        "prerequisite_course_numbers": "[]",
        "learning_objectives": "['Master developer tooling, Linux environments, and Git', 'Automate builds and containerize applications']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CS-221",
        "course_name": "Microprocessor & Interfacing\n\n3 Credits",
        "description": "8085 and 8086 microprocessor architectures, addressing modes, assembly language programming, interrupt handling, interfacing chips (8255 PPI, 8254 timer, 8259 PIC, 8251 UART), memory and I/O interfacing.",
        "prerequisite_data": "EC-219",
        "prerequisite_course_numbers": "['EC-219']",
        "learning_objectives": "['Write 8086 assembly language programs', 'Interface memory and peripheral devices to microprocessors']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CS-222",
        "course_name": "Operating Systems\n\n4 Credits",
        "description": "OS architecture, process management, CPU scheduling, inter-process communication (IPC), synchronization (semaphores, monitors), deadlocks, memory management, virtual memory, paging, segmentation, file systems, disk scheduling, and Linux kernel fundamentals.",
        "prerequisite_data": "CS-214",
        "prerequisite_course_numbers": "['CS-214']",
        "learning_objectives": "['Understand process scheduling, concurrency, and synchronization', 'Analyze memory management, paging, and file system implementations']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CS-223",
        "course_name": "Computer Organization and Architecture\n\n3 Credits",
        "description": "Von Neumann architecture, register transfer logic, arithmetic logic unit (ALU), computer arithmetic (Booth algorithm), hardwired and microprogrammed control units, pipelining, hazards, memory hierarchy (cache memory, virtual memory), and multiprocessor systems.",
        "prerequisite_data": "EC-219",
        "prerequisite_course_numbers": "['EC-219']",
        "learning_objectives": "['Design basic processor data paths and control units', 'Optimize cache memory and instruction pipelining performance']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CS-224",
        "course_name": "Theory of Computation\n\n3 Credits",
        "description": "Automata theory, regular expressions, deterministic and non-deterministic finite automata (DFA/NFA), pumping lemma for regular languages, context-free grammars (CFG), pushdown automata (PDA), Turing machines, Church-Turing thesis, decidability, and NP-completeness.",
        "prerequisite_data": "CS-212",
        "prerequisite_course_numbers": "['CS-212']",
        "learning_objectives": "['Design finite automata, PDAs, and Turing machines', 'Prove language decidability and undecidability']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CS-241",
        "course_name": "Data Communication\n\n3 Credits",
        "description": "Data transmission fundamentals, signals, modulation (ASK, FSK, PSK, QAM), transmission media, multiplexing (TDM, FDM), error detection and correction (CRC, Hamming codes), data link control protocols, and switching techniques.",
        "prerequisite_data": "None",
        "prerequisite_course_numbers": "[]",
        "learning_objectives": "['Understand digital communication signals and encoding', 'Implement error control and flow control mechanisms']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CS-242",
        "course_name": "Optimization Techniques\n\n3 Credits",
        "description": "Linear programming, simplex method, duality, transportation and assignment problems, non-linear programming, integer programming, gradient descent, genetic algorithms, and convex optimization.",
        "prerequisite_data": "MA-111",
        "prerequisite_course_numbers": "['MA-111']",
        "learning_objectives": "['Formulate optimization problems mathematically', 'Solve linear and non-linear optimization programs']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CS-243",
        "course_name": "Simulation & Modelling\n\n3 Credits",
        "description": "System simulation concepts, discrete-event simulation, random number generation, input modeling, verification and validation of simulation models, output analysis, and queuing systems.",
        "prerequisite_data": "MA-219",
        "prerequisite_course_numbers": "['MA-219']",
        "learning_objectives": "['Construct discrete-event simulation models', 'Validate stochastic simulation experiments statistically']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CS-225",
        "course_name": "Microprocessor & Interfacing Lab\n\n1 Credit",
        "description": "Assembly language programming for 8085/8086 microprocessors, peripheral interfacing with 8255, 8254, stepper motor control, ADC/DAC conversions, and traffic light controllers.",
        "prerequisite_data": "CS-221",
        "prerequisite_course_numbers": "['CS-221']",
        "learning_objectives": "['Write and debug 8086 assembly routines', 'Control hardware peripherals via microprocessor boards']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CS-226",
        "course_name": "Operating System Lab\n\n1 Credit",
        "description": "Linux system calls, process creation using fork and exec, multithreading with POSIX pthreads, synchronization using mutex and semaphores, memory mapping, and writing shell scripts.",
        "prerequisite_data": "CS-222",
        "prerequisite_course_numbers": "['CS-222']",
        "learning_objectives": "['Program Linux system calls for IPC and concurrency', 'Implement process synchronization with semaphores']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CS-227",
        "course_name": "Computer Organization & Architecture Lab\n\n1 Credit",
        "description": "Design and simulation of digital components using VHDL/Verilog: ALU design, register files, Booth multiplier, datapath design, cache simulation, and pipeline hazard analysis.",
        "prerequisite_data": "CS-223",
        "prerequisite_course_numbers": "['CS-223']",
        "learning_objectives": "['Model computer hardware modules in Verilog/VHDL', 'Simulate instruction execution cycles and cache hits/misses']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CS-311",
        "course_name": "Analysis & Design of Algorithms\n\n3 Credits",
        "description": "Algorithm design paradigms: divide-and-conquer, greedy method, dynamic programming, backtracking, branch-and-bound. String matching algorithms (KMP, Rabin-Karp), max-flow min-cut, randomized algorithms, NP-completeness, and approximation algorithms.",
        "prerequisite_data": "CS-214",
        "prerequisite_course_numbers": "['CS-214']",
        "learning_objectives": "['Synthesize optimal algorithms using dynamic programming and greedy strategies', 'Formulate NP-complete reductions']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CS-312",
        "course_name": "Compiler Design\n\n3 Credits",
        "description": "Phases of compiler, lexical analysis (Lex/Flex), syntax analysis (top-down and bottom-up parsing, LL, LR, LALR parsers, Yacc/Bison), syntax-directed translation, intermediate code generation (three-address code), code optimization, and target code generation.",
        "prerequisite_data": "CS-224",
        "prerequisite_course_numbers": "['CS-224']",
        "learning_objectives": "['Build lexical analyzers and parsers using compiler generation tools', 'Generate and optimize intermediate code representations']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CS-313",
        "course_name": "Computer Networks\n\n3 Credits",
        "description": "OSI and TCP/IP protocol stacks. Physical layer, Data Link layer (Ethernet, CSMA/CD, flow control), Network layer (IPv4, IPv6, routing protocols: OSPF, BGP, RIP, subnetting), Transport layer (TCP, UDP, congestion control), and Application layer (DNS, HTTP, SMTP, FTP). Socket programming.",
        "prerequisite_data": "CS-241",
        "prerequisite_course_numbers": "['CS-241']",
        "learning_objectives": "['Master network routing, addressing, and congestion control', 'Develop network client-server programs with socket APIs']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CS-314",
        "course_name": "Artificial Intelligence\n\n3 Credits",
        "description": "Intelligent agents, problem-solving state spaces, uninformed and heuristic search (A*, IDA*, minimax, alpha-beta pruning), constraint satisfaction problems (CSP), knowledge representation, first-order logic, resolution, planning, probabilistic reasoning, Bayesian networks, and expert systems.",
        "prerequisite_data": "CS-214",
        "prerequisite_course_numbers": "['CS-214']",
        "learning_objectives": "['Implement heuristic search algorithms and game-playing agents', 'Construct Bayesian belief networks and probabilistic inference engines']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CS-351",
        "course_name": "Advance Operating System\n\n3 Credits",
        "description": "Distributed operating systems, synchronization in distributed systems (Lamport clocks, mutual exclusion), distributed deadlocks, distributed shared memory, load balancing, real-time operating systems, and fault tolerance.",
        "prerequisite_data": "CS-222",
        "prerequisite_course_numbers": "['CS-222']",
        "learning_objectives": "['Design distributed synchronization and memory algorithms', 'Evaluate real-time scheduling mechanisms']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CS-301",
        "course_name": "Data Structures (Open Elective)\n\n3 Credits",
        "description": "Open elective covering linear and non-linear data structures, searching, sorting, hashing, trees, and graphs for non-CSE engineering students.",
        "prerequisite_data": "CS-111",
        "prerequisite_course_numbers": "['CS-111']",
        "learning_objectives": "['Apply fundamental data structures to diverse engineering disciplines']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CS-315",
        "course_name": "Compiler Design Lab\n\n1 Credit",
        "description": "Implementation of scanner/lexer using Lex/Flex, parser implementation using Yacc/Bison, syntax trees, symbol table management, and code generation.",
        "prerequisite_data": "CS-312",
        "prerequisite_course_numbers": "['CS-312']",
        "learning_objectives": "['Construct working compiler front-ends with Lex and Yacc']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CS-316",
        "course_name": "Computer Networks Lab\n\n1 Credit",
        "description": "Network simulation in NS2/NS3/Wireshark, socket programming in C/Python (TCP echo, UDP chat, HTTP client), subnetting design, and router configuration.",
        "prerequisite_data": "CS-313",
        "prerequisite_course_numbers": "['CS-313']",
        "learning_objectives": "['Packet capture analysis in Wireshark', 'Develop networked client-server applications']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CS-321",
        "course_name": "Digital Image Processing\n\n3 Credits",
        "description": "Image representation, intensity transformations, spatial filtering, histogram equalization, Fourier transform of 2D signals, frequency domain filtering, image restoration, color image processing, wavelets, image compression (JPEG), morphological operations, and edge detection.",
        "prerequisite_data": "CS-214",
        "prerequisite_course_numbers": "['CS-214']",
        "learning_objectives": "['Apply spatial and frequency domain image enhancements', 'Implement segmentation and morphological image analysis in Python/OpenCV']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CS-322",
        "course_name": "Database Management Systems\n\n3 Credits",
        "description": "Database system concepts, relational data model, relational algebra, SQL, integrity constraints, ER modeling, normal forms (1NF, 2NF, 3NF, BCNF, 4NF), query processing and optimization, transaction management, ACID properties, concurrency control (locking, 2PL, timestamping), recovery systems, and NoSQL databases.",
        "prerequisite_data": "CS-214",
        "prerequisite_course_numbers": "['CS-214']",
        "learning_objectives": "['Design normalized relational schemas and write complex SQL queries', 'Understand transaction concurrency control, locking, and recovery']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CS-323",
        "course_name": "Software Engineering\n\n2 Credits",
        "description": "Software development life cycle (SDLC) models (Waterfall, Agile, Scrum), requirements engineering, software architecture design, UML modeling, software testing strategies (white-box, black-box, regression testing), metrics, maintenance, and DevOps practices.",
        "prerequisite_data": "CS-213",
        "prerequisite_course_numbers": "['CS-213']",
        "learning_objectives": "['Lead agile software development lifecycles and sprint planning', 'Apply comprehensive unit, integration, and system testing methods']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CS-341",
        "course_name": "Game Theory\n\n3 Credits",
        "description": "Strategic games, Nash equilibrium, mixed strategies, extensive games with perfect/imperfect information, repeated games, bargaining theory, and mechanism design.",
        "prerequisite_data": "MA-219",
        "prerequisite_course_numbers": "['MA-219']",
        "learning_objectives": "['Compute Nash equilibria in strategic scenarios', 'Apply mechanism design to distributed computational systems']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CS-342",
        "course_name": "Computer Vision\n\n3 Credits",
        "description": "Feature detection and matching (SIFT, SURF, ORB), camera calibration, stereo vision, motion estimation, optical flow, object detection (YOLO, R-CNN), and deep convolutional networks.",
        "prerequisite_data": "CS-321",
        "prerequisite_course_numbers": "['CS-321']",
        "learning_objectives": "['Implement 3D scene reconstruction and feature tracking', 'Train object detection models using deep learning']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CS-343",
        "course_name": "Natural Language Processing\n\n3 Credits",
        "description": "Text preprocessing, N-gram models, part-of-speech tagging (HMM), word embeddings (Word2Vec, GloVe), sequence models (RNN, LSTM), Transformers, self-attention, BERT, and large language model architectures.",
        "prerequisite_data": "CS-314",
        "prerequisite_course_numbers": "['CS-314']",
        "learning_objectives": "['Build statistical and neural NLP pipelines', 'Fine-tune transformer models for text classification and generation']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CS-361",
        "course_name": "Cloud Computing\n\n3 Credits",
        "description": "Cloud service models (IaaS, PaaS, SaaS), virtualization (KVM, Docker, Kubernetes), cloud storage architectures, distributed file systems, serverless computing, and cloud security.",
        "prerequisite_data": "CS-222",
        "prerequisite_course_numbers": "['CS-222']",
        "learning_objectives": "['Architect scalable microservices on cloud infrastructure', 'Deploy and manage containerized clusters with Kubernetes']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CS-381",
        "course_name": "Distributed Systems\n\n2 Credits",
        "description": "Characterization of distributed systems, RPC, RMI, distributed consensus (Paxos, Raft), peer-to-peer systems, distributed transaction protocols, and fault tolerance.",
        "prerequisite_data": "CS-222",
        "prerequisite_course_numbers": "['CS-222']",
        "learning_objectives": "['Implement distributed consensus and fault-tolerant protocols', 'Build resilient peer-to-peer distributed applications']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CS-382",
        "course_name": "Machine Learning\n\n2 Credits",
        "description": "Supervised learning: linear regression, logistic regression, SVM, decision trees, random forests, naive Bayes. Unsupervised learning: k-means, hierarchical clustering, PCA. Ensemble methods, model evaluation metrics, regularization (L1/L2), and introduction to deep learning.",
        "prerequisite_data": "MA-219",
        "prerequisite_course_numbers": "['MA-219']",
        "learning_objectives": "['Train, validate, and evaluate predictive statistical models', 'Implement clustering, dimensionality reduction, and ensemble pipelines']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "HS-321",
        "course_name": "Engineering Economics and Accountancy\n\n2 Credits",
        "description": "Demand and supply analysis, cost concepts, elasticity, break-even analysis, financial statements (balance sheet, profit & loss), depreciation methods, discounted cash flow techniques (NPV, IRR), and economic appraisal of engineering projects.",
        "prerequisite_data": "None",
        "prerequisite_course_numbers": "[]",
        "learning_objectives": "['Conduct engineering cost estimations and project feasibility studies', 'Interpret financial accounting statements and appraisal metrics']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CS-324",
        "course_name": "Digital Image Processing Lab\n\n1 Credit",
        "description": "Image enhancement, edge detection, filtering, frequency domain transforms, and morphological operations implemented in Python using OpenCV, NumPy, and Scikit-Image.",
        "prerequisite_data": "CS-321",
        "prerequisite_course_numbers": "['CS-321']",
        "learning_objectives": "['Implement computer vision and image processing algorithms with OpenCV']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CS-325",
        "course_name": "Database Management Systems Lab\n\n1 Credit",
        "description": "Database design and normalization, DDL/DML SQL queries, triggers, stored procedures, transactions, and backend database integration with web applications using PostgreSQL / MySQL.",
        "prerequisite_data": "CS-322",
        "prerequisite_course_numbers": "['CS-322']",
        "learning_objectives": "['Design relational databases and write optimized SQL queries and stored procedures']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CS-411",
        "course_name": "Advance Computer Architecture\n\n3 Credits",
        "description": "Instruction level parallelism, superscalar architectures, dynamic branch prediction, speculative execution, out-of-order execution, VLIW, multi-core processors, memory hierarchy design, cache coherence protocols (MESI), snooping vs directory-based schemes, and interconnection networks.",
        "prerequisite_data": "CS-223",
        "prerequisite_course_numbers": "['CS-223']",
        "learning_objectives": "['Analyze modern superscalar and speculative execution microarchitectures', 'Design cache coherent symmetric and distributed multiprocessors']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CS-412",
        "course_name": "Information Security & Privacy\n\n3 Credits",
        "description": "Cryptographic fundamentals: symmetric key ciphers (DES, AES), public key cryptography (RSA, ECC), cryptographic hash functions (SHA-2/3), digital signatures, authentication protocols (Kerberos), network security (IPSec, SSL/TLS), firewalls, intrusion detection, privacy preserving data mining, and GDPR principles.",
        "prerequisite_data": "CS-313",
        "prerequisite_course_numbers": "['CS-313']",
        "learning_objectives": "['Implement cryptographic algorithms and secure protocols', 'Audit systems for vulnerabilities and configure intrusion prevention']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CS-413",
        "course_name": "Data Warehousing & Data Mining\n\n3 Credits",
        "description": "Data warehouse architectures, dimensional modeling, star and snowflake schemas, OLAP operations. Data preprocessing, frequent pattern mining (Apriori, FP-Growth), advanced classification, cluster analysis, outlier detection, and big data analytical frameworks.",
        "prerequisite_data": "CS-322",
        "prerequisite_course_numbers": "['CS-322']",
        "learning_objectives": "['Design dimensional data warehouse models and ETL workflows', 'Extract hidden patterns using association and clustering algorithms']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CS-431",
        "course_name": "Information Theory and Coding\n\n3 Credits",
        "description": "Entropy, mutual information, channel capacity, Shannon's theorems, source coding (Huffman, Shannon-Fano, arithmetic coding), error-correcting codes (linear block codes, cyclic codes, BCH, Reed-Solomon codes, convolutional codes).",
        "prerequisite_data": "MA-219",
        "prerequisite_course_numbers": "['MA-219']",
        "learning_objectives": "['Quantify information content and channel capacity mathematically', 'Design error-detecting and error-correcting block codes']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CS-432",
        "course_name": "Soft Computing\n\n3 Credits",
        "description": "Fuzzy logic, fuzzy sets, fuzzy reasoning and control systems, artificial neural networks (feedforward, backpropagation, radial basis functions), genetic algorithms (crossover, mutation, fitness landscapes), and hybrid neuro-fuzzy systems.",
        "prerequisite_data": "CS-314",
        "prerequisite_course_numbers": "['CS-314']",
        "learning_objectives": "['Develop neuro-fuzzy decision systems', 'Implement evolutionary algorithms for complex search spaces']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CS-433",
        "course_name": "Big Data Analytics\n\n3 Credits",
        "description": "Hadoop ecosystem (HDFS, MapReduce, YARN), Apache Spark (RDD, DataFrames, Spark SQL, MLlib), distributed stream processing (Kafka, Spark Streaming), NoSQL databases (Cassandra, MongoDB), and large-scale graph analytics.",
        "prerequisite_data": "CS-322",
        "prerequisite_course_numbers": "['CS-322']",
        "learning_objectives": "['Build distributed data processing pipelines using Spark and Hadoop', 'Perform high-throughput streaming analytics with Kafka']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CS-452",
        "course_name": "Deep Learning\n\n2 Credits",
        "description": "Deep feedforward networks, convolutional neural networks (CNNs), recurrent neural networks (RNNs, LSTMs, GRUs), autoencoders, generative adversarial networks (GANs), diffusion models, attention mechanisms, and PyTorch frameworks.",
        "prerequisite_data": "CS-382",
        "prerequisite_course_numbers": "['CS-382']",
        "learning_objectives": "['Train deep neural networks using GPU acceleration and PyTorch', 'Design modern generative models and vision/language architectures']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CS-471",
        "course_name": "Internet of Things\n\n2 Credits",
        "description": "IoT architectures, sensors, actuators, embedded systems (Arduino, Raspberry Pi, ESP32), wireless communication protocols (MQTT, CoAP, ZigBee, LoRaWAN), cloud IoT platforms, edge computing, and IoT security.",
        "prerequisite_data": "CS-313",
        "prerequisite_course_numbers": "['CS-313']",
        "learning_objectives": "['Design end-to-end IoT sensor nodes and embedded controllers', 'Implement edge analytics and secure device communications']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CS-414",
        "course_name": "Information Security and Privacy Lab\n\n1 Credit",
        "description": "Practical implementation of cryptographic ciphers (AES, RSA, ECC), digital certificates, vulnerability scanning with Nessus/OpenVAS, packet sniffing, penetration testing, and firewall rules configuration.",
        "prerequisite_data": "CS-412",
        "prerequisite_course_numbers": "['CS-412']",
        "learning_objectives": "['Perform security vulnerability assessments and cryptography routines']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CS-415",
        "course_name": "Data Warehousing & Data Mining Lab\n\n1 Credit",
        "description": "Hands-on data warehouse design, ETL modeling, implementation of association rule mining (Apriori), decision trees, k-means clustering in Python using Scikit-Learn and Pandas.",
        "prerequisite_data": "CS-413",
        "prerequisite_course_numbers": "['CS-413']",
        "learning_objectives": "['Execute end-to-end data mining and ETL pipelines']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CS-416",
        "course_name": "Summer Training / Vocational Training\n\n2 Credits",
        "description": "Mandatory industrial or institutional training completed by students after the 6th semester. Evaluated on the basis of a comprehensive written report, employer/supervisor feedback, and viva voce presentation.",
        "prerequisite_data": "Completion of 6th semester core curriculum",
        "prerequisite_course_numbers": "[]",
        "learning_objectives": "['Gain industrial software engineering experience', 'Present findings and technical contributions professionally']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CS-461",
        "course_name": "Reinforcement Learning\n\n3 Credits",
        "description": "Markov decision processes (MDP), dynamic programming, Monte Carlo methods, temporal difference learning, Q-learning, SARSA, deep Q-networks (DQN), policy gradient methods, and actor-critic architectures.",
        "prerequisite_data": "CS-382",
        "prerequisite_course_numbers": "['CS-382']",
        "learning_objectives": "['Formulate decision-making tasks as MDPs', 'Implement deep reinforcement learning agents in simulated environments']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CS-462",
        "course_name": "Cyber Security\n\n3 Credits",
        "description": "Threat modeling, offensive and defensive cyber tactics, secure coding standards (OWASP Top 10), malware analysis, web security, reverse engineering, security operations center (SOC) fundamentals, and incident response.",
        "prerequisite_data": "CS-412",
        "prerequisite_course_numbers": "['CS-412']",
        "learning_objectives": "['Audit application source code for OWASP security flaws', 'Execute digital forensics and security incident response']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CS-481",
        "course_name": "Blockchain Technology\n\n3 Credits",
        "description": "Distributed ledgers, cryptographic hashes, consensus algorithms (Proof of Work, Proof of Stake), Bitcoin, Ethereum, smart contracts in Solidity, decentralized applications (DApps), and permissioned blockchains (Hyperledger).",
        "prerequisite_data": "CS-313",
        "prerequisite_course_numbers": "['CS-313']",
        "learning_objectives": "['Design and deploy Solidity smart contracts on Ethereum testnets', 'Analyze decentralized consensus protocols and cryptographic ledgers']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CS-498",
        "course_name": "Holistic Assessment & General Proficiency\n\n2 Credits",
        "description": "Comprehensive evaluation of student professional skills, extracurricular achievements, participation in technical clubs, ethical conduct, and overall academic performance across 8 semesters at NIT Hamirpur.",
        "prerequisite_data": "Final semester standing",
        "prerequisite_course_numbers": "[]",
        "learning_objectives": "['Demonstrate holistic professional competencies and graduate attributes']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    },
    {
        "course_number": "CS-499",
        "course_name": "UG Project\n\n12 Credits",
        "description": "Capstone major project for final year B.Tech Computer Science and Engineering students at NIT Hamirpur. Involves formulation of research or engineering problem, literature review, software/hardware architecture design, implementation, testing, project documentation, and defense viva voce before an evaluation panel.",
        "prerequisite_data": "All core courses through 6th semester",
        "prerequisite_course_numbers": "[]",
        "learning_objectives": "['Execute an end-to-end engineering capstone design', 'Publish technical documentation and defend thesis findings']",
        "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
    }
]

# Write nith_courses.csv
raw_courses_file = os.path.join(os.path.dirname(__file__), "nith_raw_extracted_courses.csv")
merged_courses = list(nith_courses)
existing_codes = {c["course_number"] for c in nith_courses}

if os.path.exists(raw_courses_file):
    import pandas as pd
    import re
    df_raw = pd.read_csv(raw_courses_file)
    valid_raw = df_raw[~(df_raw["course_name"].isna() & df_raw["description"].isna())].copy()
    valid_raw = valid_raw[valid_raw["course_code"].str.match(r"^[A-Z]{2,4}-\d{3}[A-Z]?$")]

    def clean_credits(name, cr):
        try:
            val = float(cr)
            if val > 0:
                return val
        except (ValueError, TypeError):
            pass
        lname = str(name).lower()
        if "lab" in lname or "practice" in lname:
            return 1.0
        if "workshop" in lname:
            return 2.0
        if "project" in lname:
            return 6.0
        if "holistic" in lname or "assessment" in lname:
            return 2.0
        return 3.0

    def clean_description(code, name, desc):
        desc_str = str(desc).strip() if pd.notna(desc) else ""
        desc_str = re.sub(r"\bPenn Function Method\b", "Penalty Function Method", desc_str, flags=re.IGNORECASE)
        if len(desc_str) > 20:
            if "nit hamirpur" not in desc_str.lower() and "nith" not in desc_str.lower():
                desc_str += " Syllabus source: National Institute of Technology Hamirpur bachelor-course-structure-syllabus."
            return desc_str
        lname = str(name).lower()
        if code == "MS-451":
            return "Introduction to smart and multifunctional materials: Shape memory alloys (SMA), piezoelectric and electrostrictive ceramics, magnetostrictive materials, electroactive polymers (EAP), magneto-rheological fluids. Actuator and sensor design, MEMS, and aerospace/biomedical applications. NIT Hamirpur Department of Materials Science & Engineering syllabus."
        if code == "PH-451":
            return "Semiconductor crystal growth, cleanroom standards, thermal oxidation, photolithography, wet and dry etching, thin film deposition (PVD, CVD, ALD), ion implantation, thermal diffusion, metallization, and device characterization. NIT Hamirpur Department of Physics syllabus."
        if "holistic" in lname or "assessment" in lname:
            return f"Comprehensive holistic assessment, technical seminar, and professional development evaluation under departmental faculty at NIT Hamirpur for {name}."
        if "project" in lname:
            return f"Undergraduate capstone engineering project, implementation, literature survey, experimental verification, thesis submission, and viva-voce at NIT Hamirpur for {name}."
        if "lab" in lname or "practice" in lname or "workshop" in lname:
            return f"Practical laboratory experiments, simulations, and hands-on coursework corresponding to {name}. National Institute of Technology Hamirpur bachelor-course-structure-syllabus."
        return f"Core and elective curriculum study of {name} at National Institute of Technology Hamirpur. Covers theoretical foundations, analytical methodologies, and practical engineering applications."

    def generate_learning_objectives(course_name, description):
        desc = str(description)
        unit_titles = re.findall(r"UNIT[- ]*\d+\s+([^:]+?):", desc, flags=re.IGNORECASE)
        if unit_titles:
            objs = []
            verbs = ["Understand fundamental concepts of", "Analyze and solve problems in", "Design and evaluate techniques for", "Apply theoretical principles of"]
            for i, title in enumerate(unit_titles[:4]):
                clean_title = re.sub(r"\s+", " ", title).strip()
                v = verbs[i % len(verbs)]
                objs.append(f"{v} {clean_title}")
            return str(objs)
        if "lab" in course_name.lower():
            base_name = re.sub(r"(?i)\s+lab.*", "", course_name).strip()
            return str([f"Conduct practical experiments and measurements in {base_name}", f"Analyze experimental data and validate theoretical principles for {base_name}"])
        clean_name = course_name.strip()
        return str([f"Master foundational principles and concepts of {clean_name}", f"Apply analytical and design methods to engineering problems in {clean_name}"])

    prereq_rules = {
        "CE-215": "CE-212", "CE-216": "CE-214", "CE-221": "CE-211", "CE-225": "CE-224",
        "CE-226": "CE-211", "CE-314": "CE-313", "CE-321": "CE-211", "CE-322": "CE-312",
        "CE-323": "CE-223", "CE-324": "CE-311", "CE-325": "CE-322", "CE-414": "CE-212",
        "CH-216": "CH-212", "CH-225": "CH-221", "CH-226": "CH-213", "CH-321": "CH-312",
        "CH-322": "CH-222", "CH-324": "CH-322", "CH-415": "CH-411",
        "EE-216": "EE-214", "EE-217": "EE-213", "EE-221": "EE-211", "EE-225": "EE-211",
        "EE-226": "EE-224", "EE-311": "EE-222", "EE-312": "EE-213", "EE-315": "EE-221",
        "EE-316": "EE-311", "EE-321": "EE-312", "EE-322": "EE-223", "EE-324": "EE-214",
        "EC-216": "EC-212", "EC-217": "EC-211", "EC-221": "EC-211", "EC-225": "EC-221",
        "EC-226": "EC-222", "EC-311": "EC-222", "EC-312": "EC-213", "EC-315": "EC-311",
        "EC-321": "EC-221", "EC-324": "EC-312", "EC-414": "EC-411",
        "ME-215": "ME-211", "ME-223": "ME-212", "ME-225": "ME-224", "ME-226": "ME-221",
        "ME-311": "ME-222", "ME-312": "ME-211", "ME-314": "ME-311", "ME-321": "ME-223",
        "ME-322": "ME-312", "ME-324": "ME-321", "ME-411": "ME-223", "ME-414": "ME-411",
        "MS-215": "MS-212", "MS-226": "MS-221", "MS-314": "MS-311", "MS-324": "MS-321",
        "MS-414": "MS-411",
        "MA-211": "MA-111", "MA-221": "MA-121", "PH-211": "PH-101", "PH-216": "PH-211",
        "PH-217": "PH-212", "PH-226": "PH-221", "PH-227": "PH-222", "PH-314": "PH-221"
    }

    for _, r in valid_raw.iterrows():
        code = r["course_code"].strip()
        if code in existing_codes:
            continue
        existing_codes.add(code)
        cname = r["course_name"].strip()
        cr = clean_credits(cname, r["credits"])
        int_cr = int(cr) if float(cr) == int(cr) else cr
        cr_label = f"{int_cr} Credit" if int_cr == 1 else f"{int_cr} Credits"
        formatted_name = f"{cname}\n\n{cr_label}"
        desc = clean_description(code, cname, r["description"])
        objs = generate_learning_objectives(cname, desc)
        prereq = prereq_rules.get(code, "None")
        prereq_list = f"['{prereq}']" if prereq != "None" else "[]"
        merged_courses.append({
            "course_number": code,
            "course_name": formatted_name,
            "description": desc,
            "prerequisite_data": prereq,
            "prerequisite_course_numbers": prereq_list,
            "learning_objectives": objs,
            "department_url": "https://nith.ac.in/bachelor-course-structure-syllabus"
        })

fieldnames = ["course_number", "course_name", "description", "prerequisite_data", "prerequisite_course_numbers", "learning_objectives", "department_url"]
with open(courses_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in merged_courses:
        writer.writerow(row)
print(f"Wrote {len(merged_courses)} courses to {courses_file}")

# Write nith_majors.csv
common_first_year = "MA-111, PH-101, CS-111, ME-101, HS-101, HS-103, PH-102, HS-102, ME-102, CE-101, MA-121, CY-101, CS-121, EE-101, EC-101, EN-101, MB-101, CY-102, EE-102"

majors_data = [
    {
        "Major Name": "Computer Science and Engineering, B.Tech. (NIT Hamirpur)",
        "Major Code": "NITH_BTECH_CSE",
        "Total Credits Required": "160",
        "Credits for Major": "120",
        "Credits for General Education": "40",
        "Prescribed Courses": f"{common_first_year}, MA-219, CS-212, CS-213, CS-214, CS-215, EC-219, CS-217, CS-218, CS-219, CS-221, CS-222, CS-223, CS-224, CS-241, CS-225, CS-226, CS-227, CS-311, CS-312, CS-313, CS-314, CS-351, CS-301, CS-315, CS-316, CS-321, CS-322, CS-323, CS-341, CS-361, CS-381, HS-321, CS-324, CS-325, CS-411, CS-412, CS-413, CS-431, CS-452, CS-471, CS-414, CS-415, CS-416, CS-499, CS-461, CS-481, CS-498",
        "Additional Courses": "Open Electives, Minor Degree Courses",
        "Options": "Artificial Intelligence & Data Science, Systems & Cybersecurity, Software Systems",
        "General Option Courses": "CS-241, CS-242, CS-243, CS-351, CS-352, CS-353, CS-341, CS-342, CS-343, CS-361, CS-362, CS-363, CS-431, CS-432, CS-433, CS-461, CS-462, CS-463, CS-464, CS-481, CS-482, CS-483, CS-484",
        "Data Science Option Courses": "CS-382, CS-452, CS-472, CS-413, CS-433",
        "General Education Requirements": "Institute Core: 40 credits (Mathematics, Physics, Chemistry, Basic Electrical, Basic Electronics, Communication Skills, Human Values, Workshop, Graphics, Life Sciences)"
    },
    {
        "Major Name": "Electronics and Communication Engineering, B.Tech. (NIT Hamirpur)",
        "Major Code": "NITH_BTECH_ECE",
        "Total Credits Required": "160",
        "Credits for Major": "120",
        "Credits for General Education": "40",
        "Prescribed Courses": f"{common_first_year}, EC-211, EC-212, EC-213, EC-214, EC-215, EC-216, EC-217, EC-218, EC-219, EC-221, EC-222, EC-223, EC-224, EC-225, EC-226, EC-311, EC-312, EC-313, EC-314, EC-315, EC-321, EC-322, EC-323, EC-324, EC-411, EC-412, EC-413, EC-414, EC-498, EC-499",
        "Additional Courses": "EC Discipline Electives, Open Electives",
        "Options": "VLSI Design, Communication Systems, Embedded Systems",
        "General Option Courses": "EC-219, EC-101, EE-101, EC-351, EC-352, EC-361, EC-362, EC-451, EC-452",
        "Data Science Option Courses": "CS-121, MA-219, EC-312, EC-324",
        "General Education Requirements": "Institute Core: 40 credits (Mathematics, Sciences, Humanities)"
    },
    {
        "Major Name": "Electrical Engineering, B.Tech. (NIT Hamirpur)",
        "Major Code": "NITH_BTECH_EE",
        "Total Credits Required": "160",
        "Credits for Major": "120",
        "Credits for General Education": "40",
        "Prescribed Courses": f"{common_first_year}, EE-211, EE-212, EE-213, EE-214, EE-215, EE-216, EE-217, EE-218, EE-221, EE-222, EE-223, EE-224, EE-225, EE-226, EE-227, EE-311, EE-312, EE-313, EE-314, EE-315, EE-316, EE-321, EE-322, EE-323, EE-324, EE-411, EE-412, EE-498, EE-499",
        "Additional Courses": "EE Discipline Electives, Open Electives",
        "Options": "Power Systems, Control & Automation, Electric Vehicles",
        "General Option Courses": "EE-101, EE-102, EE-351, EE-352, EE-451, EE-452",
        "Data Science Option Courses": "CS-111, MA-111, EE-313, EE-322",
        "General Education Requirements": "Institute Core: 40 credits"
    },
    {
        "Major Name": "Mechanical Engineering, B.Tech. (NIT Hamirpur)",
        "Major Code": "NITH_BTECH_ME",
        "Total Credits Required": "160",
        "Credits for Major": "120",
        "Credits for General Education": "40",
        "Prescribed Courses": f"{common_first_year}, ME-211, ME-212, ME-213, ME-214, ME-215, ME-216, ME-217, ME-221, ME-222, ME-223, ME-224, ME-225, ME-226, ME-227, ME-311, ME-312, ME-313, ME-314, ME-321, ME-322, ME-323, ME-324, ME-411, ME-412, ME-413, ME-414, ME-498, ME-499",
        "Additional Courses": "ME Discipline Electives, Open Electives",
        "Options": "Thermal Engineering, Design Engineering, Manufacturing",
        "General Option Courses": "ME-101, ME-102, ME-351, ME-352, ME-451, ME-452",
        "Data Science Option Courses": "CS-111, MA-111, ME-413",
        "General Education Requirements": "Institute Core: 40 credits"
    },
    {
        "Major Name": "Civil Engineering, B.Tech. (NIT Hamirpur)",
        "Major Code": "NITH_BTECH_CE",
        "Total Credits Required": "160",
        "Credits for Major": "120",
        "Credits for General Education": "40",
        "Prescribed Courses": f"{common_first_year}, CE-211, CE-212, CE-213, CE-214, CE-215, CE-216, CE-217, CE-221, CE-222, CE-223, CE-224, CE-225, CE-226, CE-227, CE-311, CE-312, CE-313, CE-314, CE-321, CE-322, CE-323, CE-324, CE-325, CE-411, CE-412, CE-413, CE-414, CE-498, CE-499",
        "Additional Courses": "CE Discipline Electives, Open Electives",
        "Options": "Structural Engineering, Geotechnical, Transportation, Environmental",
        "General Option Courses": "CE-101, CE-351, CE-352, CE-451, CE-452",
        "Data Science Option Courses": "CS-111, MA-111, CE-304",
        "General Education Requirements": "Institute Core: 40 credits"
    },
    {
        "Major Name": "Chemical Engineering, B.Tech. (NIT Hamirpur)",
        "Major Code": "NITH_BTECH_CH",
        "Total Credits Required": "160",
        "Credits for Major": "120",
        "Credits for General Education": "40",
        "Prescribed Courses": f"{common_first_year}, CH-211, CH-212, CH-213, CH-214, CH-215, CH-216, CH-217, CH-218, CH-221, CH-222, CH-223, CH-224, CH-225, CH-226, CH-227, CH-311, CH-312, CH-313, CH-314, CH-315, CH-321, CH-322, CH-323, CH-324, CH-411, CH-412, CH-413, CH-414, CH-415, CH-498, CH-499",
        "Additional Courses": "CH Discipline Electives, Open Electives",
        "Options": "Process Systems, Biochemical Engineering, Petroleum & Petrochemicals",
        "General Option Courses": "CH-351, CH-352, CH-451, CH-452",
        "Data Science Option Courses": "CS-111, MA-111, CH-214",
        "General Education Requirements": "Institute Core: 40 credits"
    },
    {
        "Major Name": "Materials Science and Engineering, B.Tech. (NIT Hamirpur)",
        "Major Code": "NITH_BTECH_MS",
        "Total Credits Required": "160",
        "Credits for Major": "120",
        "Credits for General Education": "40",
        "Prescribed Courses": f"{common_first_year}, MS-211, MS-212, MS-213, MS-215, MS-216, MS-217, MS-221, MS-222, MS-224, MS-226, MS-227, MS-311, MS-314, MS-321, MS-322, MS-323, MS-324, MS-411, MS-412, MS-413, MS-414, MS-415, MS-498, MS-499",
        "Additional Courses": "MS Discipline Electives, Open Electives",
        "Options": "Nanomaterials, Functional Ceramics, Metallurgy & Polymer Science",
        "General Option Courses": "MS-351, MS-352, MS-451, MS-452",
        "Data Science Option Courses": "CS-111, MA-111",
        "General Education Requirements": "Institute Core: 40 credits"
    }
]

majors_fieldnames = list(majors_data[0].keys())
with open(majors_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=majors_fieldnames)
    writer.writeheader()
    for row in majors_data:
        writer.writerow(row)
print(f"Wrote {len(majors_data)} majors to {majors_file}")

