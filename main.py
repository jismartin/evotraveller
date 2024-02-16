import streamlit as st
import numpy as np
from bokeh.plotting import figure
from bokeh.models import Range1d
import algorithm

st.set_page_config(page_title="Genetic algorithm for TSP", page_icon=None)

st.header('Genetic algorithm for TSP')
st.write('This is an interactive tool for visualizing the Traveling Salesman Problem (TSP) solution using a Genetic Algorithm (GA)')
st.write('Choose the parameters (sidebar) for the Genetic Algorithm and click on "Run" to visualize the solution')

st.sidebar.subheader('GA parameters')
population_size = st.sidebar.slider('Population size', min_value=10, max_value=100, value=100, step=2)
generations = st.sidebar.slider('Generations', min_value=10, max_value=100, value=50)
tournament_size = st.sidebar.slider('Tournament size', min_value=2, max_value=10, value=4)
mutation_rate = st.sidebar.slider('Mutation rate', min_value=0.0, max_value=0.2, value=0.05, step=0.01)
elitism = st.sidebar.slider('Elitism', min_value=0.0, max_value=0.5, value=0.1, step=0.1)
crossover = st.sidebar.slider('Crossover', min_value=0.0, max_value=1.0-elitism, value=0.6, step=0.1)

run=st.button(label='Run')

if 'run' not in st.session_state:
    st.session_state['run'] = False
if 'gen' not in st.session_state:
    st.session_state['gen'] = 0
if 'best_solution' not in st.session_state:
    st.session_state['best_solution'] = None
if 'fitness_evolution' not in st.session_state:
    st.session_state['fitness_evolution'] = None
if 'dfsim' not in st.session_state:
    st.session_state['dfsim'] = None    

# Define the charts
def fitness_evolution_chart(timeseries):
    x=np.arange(len(timeseries))
    y=timeseries
    p = figure(width=200, height=200, title="Fitness", tools="", x_axis_label='Generation',
           toolbar_location=None, match_aspect=True)
    p.line(x, y, color="navy", alpha=0.4, line_width=4)
    p.y_range = Range1d(min(timeseries)-0.0005, max(timeseries)+0.0005)
    p.background_fill_color = "#efefef"
    return p

def best_solution_chart(solution):
    padding=0.1
    x, y = zip(*solution)
    p = figure(width=200, height=200, title="Best solution", tools="", 
           toolbar_location=None, match_aspect=True)
    p.line(x, y, color="navy", alpha=0.4, line_width=4)
    p.x_range = Range1d(0-padding, algorithm.grid_size + padding)
    p.y_range = Range1d(0-padding, algorithm.grid_size + padding)
    p.axis.visible = False 
    p.background_fill_color = "#efefef"
    return p

def solution_population_chart(P,g):
    S=int(np.sqrt(population_size)) + 1 
    padding=0.1
    L=algorithm.grid_size + 2*padding
  
    p = figure(width=400, height=400, title="Generation %i" %g, tools="",
           toolbar_location=None, match_aspect=True)

    for i in range(S):
        H=i*L + padding 
        for j in range(S):
            W=j*L + padding
            if P:
                solution=P.pop()
                x, y = zip(*solution)
                p.line(np.array(x)+W, np.array(y)+H, color="navy", alpha=0.4, line_width=4)

    p.x_range = Range1d(0, L*S)
    p.y_range = Range1d(0, L*(S-1))
    p.axis.visible = False 
    p.background_fill_color = "#efefef"
    return p



# Run the genetic algorithm and draw the charts
if run:
    st.session_state['run'] = True
    # Run the genetic algorithm
    targets=algorithm.places(algorithm.grid_size,algorithm.n_places)
    st.session_state['best_solution'],st.session_state['fitness_evolution'], st.session_state['dfsim'] = algorithm.genetic_algorithm(
        targets, population_size,generations, tournament_size, mutation_rate, elitism,crossover)
    st.session_state['best_solution']=[(0,0)] + st.session_state['best_solution'] + [(0,0)]


 
if st.session_state['run']==True:

    col1, col2 = st.columns(2)
    
    with col1:
        # Draw the fitness evolution
        st.write('Fitness evolution')
        pchart1=fitness_evolution_chart(st.session_state['fitness_evolution'])
        st.bokeh_chart(pchart1, use_container_width=True)

    with col2:
        # Draw the best solution
        st.write('Best solution')
        pchart2=best_solution_chart(st.session_state['best_solution'])
        st.bokeh_chart(pchart2, use_container_width=True)

    # Show the solutions
    st.write('Solutions')
    st.session_state['gen']=st.slider('Generation', min_value=0, max_value=generations-1, value=0)
    pchart3 = solution_population_chart(st.session_state['dfsim'].loc[st.session_state.gen].values.tolist(),
                                             st.session_state['gen'])
    st.bokeh_chart(pchart3, use_container_width=True)


st.write('Source code and ⭐ at [GitHub](https://github.com/jismartin/evotraveller)')
st.write('Authors: The Gooniees')