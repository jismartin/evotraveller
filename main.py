import streamlit as st
import streamlit as st
import streamlit.components.v1 as components # Standard library
from bokeh.embed import file_html
from bokeh.resources import CDN
import numpy as np
from bokeh.plotting import figure
from bokeh.models import Range1d
#from streamlit_bokeh import st_bokeh_chart
import algorithm

st.set_page_config(page_title="Genetic algorithm for TSP")

# Sidebar

## TSP parameters
st.sidebar.subheader('TSP parameters')
n_places = st.sidebar.slider('Number of places', min_value=5, max_value=15, value=10)
seed=st.sidebar.number_input('Seed', min_value=0, max_value=10000, value=1111)

## GA parameters
st.sidebar.subheader('GA parameters')
population_size = st.sidebar.slider('Population size', min_value=10, max_value=100, value=100, step=2)
generations = st.sidebar.slider('Generations', min_value=10, max_value=100, value=50)
tournament_size = st.sidebar.slider('Tournament size', min_value=2, max_value=10, value=4)
mutation_rate = st.sidebar.slider('Mutation rate', min_value=0.0, max_value=0.2, value=0.05, step=0.01)
elitism = st.sidebar.slider('Elitism', min_value=0.01, max_value=0.1, value=0.05, step=0.01)
crossover = st.sidebar.slider('Crossover', min_value=0.0, max_value=1.0-elitism, value=0.6, step=0.1)

## Parameters description
st.sidebar.subheader('Description of the parameters')
st.sidebar.write("Number of places: is the number of cities that the salesman has to visit")
st.sidebar.write("Seed: is the seed for the random generation of locations of the TSP")
st.sidebar.write("Population size: is the number of potential solutions that are randomly evaluated in each generation") 
st.sidebar.write("Generations: is the maximum number of iterations that the algorithm will run for")
st.sidebar.write("Tournament size: is the number of individuals that are selected to compete in the tournament")
st.sidebar.write("Mutation: is the probability that two randomly selected places in a sequence will flip")
st.sidebar.write("Elitism: is the proportion of the best solutions that are kept for the next generation")
st.sidebar.write("Crossover: is the proportion of the population that is replaced by the offspring of the selected solutions")

# Main page
st.header('Genetic algorithm for TSP')
st.write('This is an interactive tool for visualizing the Traveling Salesman Problem (TSP) solution using a Genetic Algorithm (GA)')
st.write('Choose the parameters (sidebar) for the Genetic Algorithm and click on "Run" to visualize the solution')

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
# Fitness evolution
def fitness_evolution_chart(timeseries):
    x=np.arange(len(timeseries))
    y=timeseries
    p = figure(width=200, height=200, title="Fitness", tools="", x_axis_label='Generation',
           toolbar_location=None, match_aspect=True)
    p.line(x, y, color="navy", alpha=0.4, line_width=4)
    p.y_range = Range1d(min(timeseries)-0.0005, max(timeseries)+0.0005)
    p.background_fill_color = "#efefef"
    return p

# Best solution
def best_solution_chart(solution):
    padding=0.1
    x, y = zip(*solution)
    p = figure(width=200, height=200, title="Best solution", tools="", 
           toolbar_location=None, match_aspect=True)
    p.line(x, y, color="navy", alpha=0.4, line_width=4)
    p.circle(x, y, color='red',size=10)
    p.x_range = Range1d(0-padding, algorithm.grid_size + padding)
    p.y_range = Range1d(0-padding, algorithm.grid_size + padding)
    p.axis.visible = False 
    p.background_fill_color = "#efefef"
    return p

# Population of solutions
def solution_population_chart(P_input, g):
    # Use a copy so we don't destroy the data in session_state
    P = P_input.copy() 
    S = int(np.sqrt(population_size)) + 1 
    padding = 0.1
    L = algorithm.grid_size + 2 * padding
  
    p = figure(width=400, height=400, title=f"Generation {g}", 
               tools="pan,wheel_zoom,reset", # Added standard tools
               toolbar_location=None, match_aspect=True)

    for i in range(S):
        H = i * L + padding 
        for j in range(S):
            W = j * L + padding
            if P:
                solution = P.pop()
                x, y = zip(*solution)
                p.line(np.array(x) + W, np.array(y) + H, color="navy", alpha=0.4, line_width=2)

    p.x_range = Range1d(0, L * S)
    p.y_range = Range1d(0, L * (S - 1))
    p.axis.visible = False 
    p.background_fill_color = "#f9f9f9" # Lightened for modern look
    return p

# Create a helper function to render the Bokeh plot
def render_bokeh(plot):
    html = file_html(plot, CDN, "my plot")
    # We use a fixed height or adjust based on your plot settings
    components.html(html, height=plot.height + 20)

# Run the genetic algorithm and draw the charts
if run:
    st.session_state['run'] = True
    # Run the genetic algorithm
    algorithm.set_seed(seed)
    targets=algorithm.places(algorithm.grid_size,n_places-1)
    st.session_state['best_solution'],st.session_state['fitness_evolution'], st.session_state['dfsim'] = algorithm.genetic_algorithm(
        targets, population_size,generations, tournament_size, mutation_rate, elitism,crossover)
    st.session_state['best_solution']=[(0,0)] + st.session_state['best_solution'] + [(0,0)]
 
if st.session_state['run']==True:

    col1, col2 = st.columns(2)
    
    with col1:
        st.write('Fitness evolution')
        pchart1 = fitness_evolution_chart(st.session_state['fitness_evolution'])
        # 2. Use the helper instead of st.bokeh_chart
        render_bokeh(pchart1)
        
    with col2:
        st.write('Best solution')
        pchart2 = best_solution_chart(st.session_state['best_solution'])
        # 2. Use the helper instead of st.bokeh_chart
        render_bokeh(pchart2)
        st.write('The path starts/ends at (0,0)')

    # Show the solutions
    st.write('Solutions')
    st.session_state['gen'] = st.slider('Generation', 0, generations-1, value=0)
    
    # Crucial: copy the list so .pop() doesn't destroy your session data
    pop_list = st.session_state['dfsim'].loc[st.session_state.gen].values.tolist().copy()
    
    pchart3 = solution_population_chart(pop_list, st.session_state['gen'])
    # Use the helper instead of st.bokeh_chart
    render_bokeh(pchart3)

st.write('Source code and ⭐ at [GitHub](https://github.com/jismartin/evotraveller)')
st.write('Authors: The Goonies')