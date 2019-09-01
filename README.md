# NEC hackathon 2019
This is the repository of team **The Transporters** for the NEC smart transport hackathon 2019.
Developing a platform to facilitate smart public transport using Machine Learning

# Problem Statement
#### Transformation of public transport by using web or service applications to elevate user motivation
India is home to a large population of people. You require a medium of transport to move from one place to another. In India, the majority of people rely on their personal transport. Hence, this leads to roads congestions, various types of pollution, etc. You are provided with the benefits of public transportation. Your task is to build an application that must encourage people to use public transportation systems.
# Our Solution  - Intelligent Transport System
 **➤ One Stop Solution to Commuting - Anytime, anywhere in the city !!**

![](https://raw.githubusercontent.com/MananSoni42/NEC-hackathon/master/Assets/Idea.jpg?token=AHP3X2S26X3RUOQFT2X66725OTJHO)
### Key Features
-   Dynamically creating bus routes based on demand
-   Full integration with internal routes & public flow
-   Android app service for every citizen
-   Most Time & Cost effective
-   Dynamic Connected route Information at your Tips
-   Impactful communication

## Workflow of Intelligent Transportation System
![](https://raw.githubusercontent.com/MananSoni42/NEC-hackathon/master/Assets/Working.jpg?token=AHP3X2TPQXKZX73XDEEL5J25OTJL6)

![](https://raw.githubusercontent.com/MananSoni42/NEC-hackathon/master/Assets/Timeline.jpg?token=AHP3X2XFDTXTVW35BKKKR4S5OTJNQ)
##  App Design
![](https://raw.githubusercontent.com/MananSoni42/NEC-hackathon/master/Assets/App_n.jpg?token=AHP3X2U77WOROP34C2TM57C5OTK7E)
## How our algorithm works
##### Input
As input, we expect a list of bus-stops along with their corresponding latitudes and longitudes. Also, each bus-stop needs to have data about how many people got on/off at that stop. An example can be seen in the file: ``data/may-trimester-2017-stop-ridership-ranking-saturday-csv-9.csv``
#### Output
We suggest a set of optimal routes that maximize the total revenue as well as the total number of people served.

#### Step 1
We create a weighted undirected Graph of the given N bus-stops using the given data. The graph is such that the weight between any two bus-stops is given by (r is a hyper-parameter): 
d(p,q) if d(p,q) <=r
0,         otherwise

#### Step 2
We find K (< N) bus stops such that the distance between any point on our original graph and one of these K stops is minimized.
This problem is known as the [Capacitated facility location problem]([https://en.wikipedia.org/wiki/Facility_location_problem](https://en.wikipedia.org/wiki/Facility_location_problem)). We derive an approximate solution using Dynamic Programming.

#### Step 3
We create a fully connected undirected graph G1 with the K centers found earlier and the edges are weighted by the distance between these bus-stops.
 A directed graph G2 is also created with the K centers where each node of G2 has 2 weights:
 * in: The probability of a citizen getting on the bus at this station
 * out: The probability of a citizen getting off the bus at this station.
 
We, then, use the graph G2 to simulate a population of citizens where each citizen is assigned a source and a destination stop according to the above mentioned probabilities.

#### Step 4
We know train a Genetic algorithm to decide on the optimal routes.
Each member of this GA population consists of a set of routes along with the number of buses running on that route.

**Mutation**

The mutation operator replaces a random % of the nodes in any route with one of their closest neighbors.
![](https://raw.githubusercontent.com/MananSoni42/NEC-hackathon/master/Assets/Mutation.jpg?token=AHP3X2SMF5IREPQEADUAGL25OTO3A)

**Crossover**

The crossover operator does 2 things:
* Randomly exhange some % of routes between two members of the GA population
* Select 2 random routes, if they have 1 or more nodes in common, exchange the path between the common nodes (or till the end of the route if only 1 node is common)
![](https://raw.githubusercontent.com/MananSoni42/NEC-hackathon/master/Assets/Crossover.jpg?token=AHP3X2SBBD5JBI7MDAAZA2K5OTMXE)

  **The Fitness function**
  We monitor 3 things while calculating our fitness function
  * How much of the total capacity of each bus is fulfilled (in %)
  * How many (simulated) citizens were our buses able to serve
  * The total length of all the routes for a particular solution.

### Outcomes
We designed three seperate solutions and comapared them to the initial routes given in the dataset.
* **Original**: The routes given in the original dataset
	
	Profits recorded: - 15,500 Rs
	
	Number of people served: 1136 using 164 buses
* **Economical**: The model tries to find routes which optimize only the total revenue
	
	Profits recorded: 1,12,446 Rs
	
	Number of people served: 650 (-42%) using 51 buses
* **People centric**: Optimize the model so maximum number of people are able to commute without considering the revenue
	
	Profits recorded: 2202 Rs
	
	Number of people served: 3374 (197%) using 74
* **Resource centric**: The model tries to efficiently utilize all the resources in hand - It’s objective is somewhere between that of model 1 and model 2
	
	Profits recorded: 87000 Rs
	
	Number of people served: 2000 (76% capacity) using 65 buses

### Team members
* [Chinmay Hebbar](https://github.com/cheese-cracker) - Backend developer
* [Manan Soni](https://github.com/MananSoni42) - ML developer
* [Siddharth Singh](https://github.com/coolsidd) - ML developer / App developer
* [Sparsh Jain](https://github.com/dudesparsh) - Frontend developer
