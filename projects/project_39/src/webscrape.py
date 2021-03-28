import requests
import json
import pandas as pd

# for field of study conversion usage
semantic_fos = {
    "ALL": [],
    "Astrophysics": ["physics"],
    'Astrophysics of Galaxies': ["physics"],
    'Cosmology and Nongalactic Astrophysics': ["physics"],
    'Earth and Planetary Astrophysics': ["physics"],
    'High Energy Astrophysical Phenomena': ["physics"],
    'Instrumentation and Methods for Astrophysics': ["physics"],
    'Solar and Stellar Astrophysics': ["physics"],

    'Condensed Matter': ["physics"],
    'Disordered Systems and Neural Networks': ["biology", "physics"],
    'Materials Science': ["materials-science"],
    'Mesoscale and Nanoscale Physics': ["physics"],
    'Other Condensed Matter': ["physics"],
    'Quantum Gases': ["physics"],
    'Soft Condensed Matter': ["physics"],
    'Statistical Mechanics': ["physics"],
    'Strongly Correlated Electrons': ["physics"],
    'Superconductivity': ["physics"],

    'Computer Science': ["computer-science"],
    'Artificial Intelligence': ["computer-science", "engineering"],
    'Computation and Language': ["computer-science"],
    'Computational Complexity': ["computer-science"],
    'Computational Engineering, Finance, and Science': ["computer-science", "economics", "engineering"],
    'Computational Geometry': ["computer-science", "mathematics"],
    'Computer Science and Game Theory': ["computer-science", "economics"],
    'Computer Vision and Pattern Recognition': ["computer-science", "engineering"],
    'Computers and Society': ["computer-science", "sociology"],
    'Cryptography and Security': ["computer-science", "engineering", "mathematics", "physics"],
    'Data Structures and Algorithms': ["computer-science"],
    'Databases': ["computer-science"],
    'Digital Libraries': ["computer-science"],
    'Discrete Mathematics': ["mathematics"],
    'Distributed, Parallel, and Cluster Computing': ["computer-science"],
    'Emerging Technologies': ["computer-science"],
    'Formal Languages and Automata Theory': ["computer-science"],
    'General Literature': ["computer-science", "art"],
    'Graphics': ["computer-science", "art"],
    'Hardware Architecture': ["computer-science", "engineering"],
    'Human-Computer Interaction': ["computer-science", "biology", "psychology"],
    'Information Retrieval': ["computer-science"],
    'Information Theory': ["computer-science"],
    'Logic in Computer Science': ["computer-science", "mathematics"],
    'Machine Learning': ["computer-science", "mathematics"],
    'Mathematical Software': ["computer-science", "mathematics", "engineering"],
    'Multiagent Systems': ["computer-science", "mathematics", "engineering"],
    'Multimedia': ["computer-science", "sociology", "engineering"],
    'Networking and Internet Architecture': ["computer-science", "engineering"],
    'Neural and Evolutionary Computing': ["computer-science", "biology"],
    'Numerical Analysis': ["computer-science", "mathematics"],
    'Operating Systems': ["computer-science", "engineering"],
    'Other Computer Science': ["computer-science"],
    'Performance': ["computer-science"],
    'Programming Languages': ["computer-science"],
    'Robotics': ["computer-science", "engineering"],
    'Social and Information Networks': ["computer-science", "sociology"],
    'Software Engineering': ["computer-science", "engineering"],
    'Sound': ["computer-science", "engineering"],
    'Symbolic Computation': ["computer-science", "mathematics"],
    'Systems and Control': ["computer-science", "engineering"],

    'Economics': ["economics"],
    'Econometrics': ["economics"],
    'General Economics': ["economics"],
    'Theoretical Economics': ["economics"],

    'Electrical Engineering and Systems Science': ["engineering"],
    'Audio and Speech Processing': ["computer-science", "engineering"],
    'Image and Video Processing': ["computer-science", "engineering"],
    'Signal Processing': ["computer-science", "engineering"],
    'Systems and Control': ["computer-science", "engineering"],

    'High Energy Physics': ["physics"],
    'Experiment': ["physics"],
    'Lattice': ["physics"],
    'Phenomenology': ["physics"],
    'Theory': ["physics"],

    'Mathematics': ["mathematics"],
    'Algebraic Geometry': ["mathematics"],
    'Algebraic Topology': ["mathematics"],
    'Analysis of PDEs': ["mathematics"],
    'Category Theory': ["mathematics"],
    'Classical Analysis and ODEs': ["mathematics"],
    'Combinatorics': ["mathematics"],
    'Commutative Algebra': ["mathematics"],
    'Complex Variables': ["mathematics"],
    'Differential Geometry': ["mathematics"],
    'Dynamical Systems': ["mathematics"],
    'Functional Analysis': ["mathematics"],
    'General Mathematics': ["mathematics"],
    'General Topology': ["mathematics"],
    'Geometric Topology': ["mathematics"],
    'Group Theory': ["mathematics"],
    'History and Overview': ["mathematics"],
    'Information Theory': ["mathematics"],
    'K-Theory and Homology': ["mathematics"],
    'Logic': ["mathematics"],
    'Mathematical Physics': ["mathematics", "physics"],
    'Metric Geometry': ["mathematics"],
    'Number Theory': ["mathematics"],
    'Numerical Analysis': ["mathematics"],
    'Operator Algebras': ["mathematics"],
    'Optimization and Control': ["mathematics"],
    'Probability': ["mathematics"],
    'Quantum Algebra': ["mathematics"],
    'Representation Theory': ["mathematics"],
    'Rings and Algebras': ["mathematics"],
    'Spectral Theory': ["mathematics"],
    'Statistics Theory': ["mathematics"],
    'Symplectic Geometry': ["mathematics"],

    'Nonlinear Sciences': ["mathematics"],
    'Adaptation and Self-Organizing Systems': ["mathematics"],
    'Cellular Automata and Lattice Gases': ["mathematics"],
    'Chaotic Dynamics': ["mathematics"],
    'Exactly Solvable and Integrable Systems': ["mathematics"],
    'Pattern Formation and Solitons': ["mathematics"],

    'Physics': ["physics"],
    'Accelerator Physics': ["physics"],
    'Applied Physics': ["physics"],
    'Atmospheric and Oceanic Physics': ["physics"],
    'Atomic Physics': ["physics"],
    'Atomic and Molecular Clusters': ["physics"],
    'Biological Physics': ["physics", "biology"],
    'Chemical Physics': ["physics", "chemistry"],
    'Classical Physics': ["physics"],
    'Computational Physics': ["physics", "computer-science", "mathematics"],
    'Data Analysis, Statistics and Probability': ["physics", "computer-science", "mathematics"],
    'Fluid Dynamics': ["physics"],
    'General Physics': ["physics"],
    'Geophysics': ["physics", "geography"],
    'History and Philosophy of Physics': ["physics", "history", "philosophy"],
    'Instrumentation and Detectors': ["physics"],
    'Medical Physics': ["physics", "medicine"],
    'Optics': ["physics"],
    'Physics Education': ["physics"],
    'Physics and Society': ["physics", "sociology"],
    'Plasma Physics': ["physics"],
    'Popular Physics': ["physics"],
    'Space Physics': ["physics"],

    'Quantitative Biology': ["biology", "mathematics"],
    'Biomolecules': ["biology", "chemistry"],
    'Cell Behavior': ["biology", "chemistry"],
    'Genomics': ["biology", "chemistry"],
    'Molecular Networks': ["biology", "chemistry"],
    'Neurons and Cognition': ["biology", "chemistry"],
    'Other Quantitative Biology': ["biology", "mathematics"],
    'Populations and Evolution': ["biology", "sociology", "mathematics"],
    'Quantitative Methods': ["biology", "mathematics"],
    'Subcellular Processes': ["biology", "chemistry"],
    'Tissues and Organs': ["biology"],

    'Quantitative Finance': ["mathematics", "economics", "business"],
    'Computational Finance': ["computer-science", "economics", "business"],
    'Economics': ["mathematics", "economics"],
    'General Finance': ["mathematics", "economics"],
    'Mathematical Finance': ["mathematics", "economics"],
    'Portfolio Management': ["mathematics", "economics"],
    'Pricing of Securities': ["mathematics", "economics"],
    'Risk Management': ["mathematics", "economics"],
    'Statistical Finance': ["mathematics", "economics"],
    'Trading and Market Microstructure': ["mathematics", "economics"],

    'Statistics': ["mathematics"],
    'Applications': ["mathematics"],
    'Computation': ["mathematics", "computer-science"],
    'Machine Learning': ["mathematics", "computer-science"],
    'Methodology': ["mathematics"],
    'Other Statistics': ["mathematics"],
    'Statistics Theory': ["mathematics"]
}


def webscrape(unique_key, keywords_path, fos_path, out_path):
    print("\n")
    print(">>>>>>>>>>>>>>>>>>>>>>>> Running webscraping... <<<<<<<<<<<<<<<<<<<<<<<<<<<<")

#    fos_path = 'data/out/fos' + unique_key + '.json'
    fos_path = 'data/out/fos.json'
    try:
        # Extract keywords and fields of study
        given_fos = json.load(open(fos_path))['fos']
            # convert field of study
        fos = []
        for item in given_fos:
            fos = fos + semantic_fos[item]
    except:
        fos = ['computer-science']
    print("field of study: {}".format(fos))
    keywords = json.load(open(keywords_path))['keywords']
    print("keywords: {}".format(keywords))

    # send request to the server
    print("  => Sending requests...")
    payload = {
        "queryString":keywords,
        "page":1,
        "pageSize":10,
        "sort":"relevance",
        "authors":[],
        "coAuthors":[],
        "venues":[],
        "yearFilter":None,
        "requireViewablePdf":False,
        "publicationTypes":[],
        "externalContentTypes":[],
        "fieldsOfStudy":fos,
        "useFallbackRankerService":False,
        "useFallbackSearchCluster":False,
        "hydrateWithDdb":True,
        "includeTldrs":True,
        "performTitleMatch":True,
        "includeBadges":True
    }
    headers = {
        "Content-Type": "application/json",
        'Accept': 'application/json',
    }
    r = requests.post("https://www.semanticscholar.org/api/1/search", json.dumps(payload), headers = headers)

    print("  => Saving results...")
    # parse the results
    results = r.json()['results']

    # parse the specification of the results
    specifics = []
    counter = 0
    for item in results:
        cur = {}
        # parse useful tags
        try:
            cur['link'] = item['primaryPaperLink']['url']
        except:
            pass
        try:
            cur['title'] = item['title']['text']
        except:
            pass
        try:
            cur['authors'] = item['structuredAuthors']
        except:
            pass
        try:
            cur['abstract'] = item['paperAbstract']['text']
        except:
            pass
        try:
            cur['date'] = item['pubDate']
        except:
            pass
        # append current paper to the list of scraped papers
        specifics.append(cur)
        counter += 1
        # if the number of scraped papers meet our needs
        if len(specifics) == 5:
            break
    # write the result to a json file
    with open(out_path, 'w') as outfile:
        json.dump(specifics, outfile)
    
    print(" => Done! Webscrape result is saved as '" + out_path + "'")
    print("\n")
