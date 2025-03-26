# List of Indian states and union territories (lowercase for easier matching)
INDIAN_STATES = [
    "andhra pradesh",
    "arunachal pradesh",
    "assam",
    "bihar",
    "chhattisgarh",
    "goa",
    "gujarat",
    "haryana",
    "himachal pradesh",
    "jharkhand",
    "karnataka",
    "kerala",
    "madhya pradesh",
    "maharashtra",
    "manipur",
    "meghalaya",
    "mizoram",
    "nagaland",
    "odisha",
    "punjab",
    "rajasthan",
    "sikkim",
    "tamil nadu",
    "telangana",
    "tripura",
    "uttar pradesh",
    "uttarakhand",
    "west bengal",
    "andaman and nicobar islands",
    "chandigarh",
    "dadra and nagar haveli and daman and diu",
    "delhi",
    "jammu and kashmir",
    "ladakh",
    "lakshadweep",
    "puducherry",
]

# Major cities
INDIAN_CITIES = [
    "mumbai", "delhi", "bangalore", "hyderabad", "chennai", "kolkata",
    "pune", "ahmedabad", "jaipur", "lucknow", "kochi", "chandigarh",
    "bhopal", "indore", "nagpur", "patna", "surat", "visakhapatnam",
    "thiruvananthapuram", "bhubaneswar", "coimbatore", "mysore",
    "guwahati", "varanasi", "nashik", "agra", "vadodara", "mangalore",
    "trivandrum", "dehradun", "bengaluru", "calcutta", "bombay",
    "madras", "poona", "baroda", "cochin", "vizag", "benares"
]

# Alternate spellings or short forms of states
STATE_ALIASES = {
    "up": "uttar pradesh",
    "mp": "madhya pradesh",
    "delhi ncr": "delhi",
    "new delhi": "delhi",
    "ap": "andhra pradesh",
    "tn": "tamil nadu",
    "mh": "maharashtra",
    "j&k": "jammu and kashmir",
    "jk": "jammu and kashmir",
    "hp": "himachal pradesh",
    "wb": "west bengal",
    "uk": "uttarakhand",
    "uae": "uttarakhand",
}

# City aliases
CITY_ALIASES = {
    "bengaluru": "bangalore",
    "calcutta": "kolkata",
    "bombay": "mumbai",
    "madras": "chennai",
    "poona": "pune",
    "baroda": "vadodara",
    "cochin": "kochi",
    "vizag": "visakhapatnam",
    "benares": "varanasi",
}

# Common Indian names
INDIAN_NAMES = {
    "male": [
        "amit", "rahul", "rajesh", "vikram", "aditya", "arjun", "sanjay",
        "rishi", "karthik", "varun", "deepak", "rohan", "nikhil", "kunal",
        "vishal", "pranav", "siddharth", "aarav", "aryan", "kabir", "advait",
        "reyansh", "vihaan", "dhruv", "vivaan", "aayan", "ishaan", "krishna",
        "shiv", "yash", "virat", "sahil", "arnav", "dev"
    ],
    "female": [
        "priya", "neha", "ananya", "meera", "sneha", "kavita", "pooja",
        "divya", "nandini", "anjali", "shreya", "tanvi", "ishita", "swati",
        "shweta", "aisha", "ritika", "zara", "kiara", "mira", "avani",
        "myra", "aanya", "tara", "diya", "aaradhya", "saanvi", "anika",
        "riya", "isha", "pari", "aditi", "siya"
    ],
    "last_names": [
        "kumar", "singh", "sharma", "verma", "gupta", "patel", "reddy",
        "rao", "malhotra", "joshi", "chopra", "mehta", "shah", "kapoor",
        "iyer", "nair", "pillai", "desai", "choudhury", "banerjee"
    ]
}

# Entity patterns for name recognition
NAME_PATTERNS = [
    "my name is {name}",
    "i'm {name}",
    "i am {name}",
    "call me {name}",
    "{name} here",
    "this is {name}",
    "{name} speaking",
    "you can call me {name}",
    "people know me as {name}",
    "i go by {name}",
    "my full name is {name}",
    "hi, {name} here",
]

# Age patterns
AGE_PATTERNS = [
    "i am {age} years old",
    "i'm {age}",
    "i'm {age} years old",
    "my age is {age}",
    "{age} years",
    "i just turned {age}",
    "i will be {age} next month",
    "i'm {age} years of age",
    "age: {age}",
    "i was born {age} years ago",
    "i recently turned {age}",
    "i'm in my {age}s",
    "about {age} years old",
    "currently {age}",
    "{age} years young",
    "i completed {age} years"
]

# Gender patterns
GENDER_PATTERNS = [
    "i am a {gender}",
    "i'm a {gender}",
    "my gender is {gender}",
    "i identify as {gender}",
    "i am {gender}",
    "i'm {gender}",
    "born {gender}",
    "prefer {gender}",
    "i use {gender}",
    "{gender} gender",
    "consider myself {gender}",
    "identity: {gender}"
]

# Location patterns
LOCATION_PATTERNS = [
    "i live in {location}",
    "i am from {location}",
    "i'm from {location}",
    "my location is {location}",
    "my state is {location}",
    "i reside in {location}",
    "i stay in {location}",
    "my residence is in {location}",
    "currently in {location}",
    "based out of {location}",
    "native of {location}",
    "my hometown is {location}",
    "originally from {location}",
    "shifted to {location}",
    "living in {location}",
    "settled in {location}",
    "born and raised in {location}",
    "working in {location}",
    "moved to {location}",
    "staying at {location}",
    "home state is {location}",
    "permanent address: {location}"
]

# Training examples from entities.yaml
TRAINING_EXAMPLES = {
    "name": [
        "My name is Amit Kumar",
        "I am Priya Sharma",
        "This is Rajesh Singh",
        "You can call me Ananya",
        "Vikram here",
        "Hello, I'm Neha Gupta",
        "People know me as Rahul Verma",
        "I go by Sneha Patel",
        "Arjun Reddy is my name",
        "Call me Divya Malhotra",
        "My full name is Karthik Iyer",
        "Pooja Shah speaking",
        "Hi, Rohan Kapoor here",
        "I'm Aisha Mehta",
        "This is Vivaan Joshi"
    ],
    "age": [
        "I am 25 years old",
        "My age is 30",
        "I'm 42 years",
        "18 years",
        "I just turned 21",
        "I will be 35 next month",
        "I'm 28 years of age",
        "Age: 45",
        "I was born 19 years ago",
        "I recently turned 24",
        "I'm in my 50s",
        "About 33 years old",
        "Currently 27",
        "65 years young",
        "I completed 40 years"
    ],
    "gender": [
        "I am male",
        "My gender is female",
        "I identify as non-binary",
        "I'm a boy",
        "She/her pronouns",
        "I'm gender fluid",
        "Identify as transgender",
        "I'm a woman",
        "Born male",
        "Prefer they/them",
        "I use he/him",
        "Female gender",
        "I'm agender",
        "Consider myself genderqueer",
        "Identity: trans woman"
    ],
    "location": [
        "I live in Delhi",
        "I'm from Uttar Pradesh",
        "My residence is in Gujarat",
        "I stay in Mumbai",
        "Currently in Karnataka",
        "Based out of Chennai",
        "Native of Kerala",
        "Residing in Pune",
        "Located in Hyderabad",
        "My hometown is Jaipur",
        "Originally from West Bengal",
        "Shifted to Bangalore",
        "Living in Kolkata",
        "Settled in Maharashtra",
        "Born and raised in Punjab",
        "Working in Ahmedabad",
        "Moved to Chandigarh",
        "Staying at Lucknow",
        "Home state is Rajasthan",
        "Permanent address: Tamil Nadu"
    ]
} 