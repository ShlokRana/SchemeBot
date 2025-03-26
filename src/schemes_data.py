"""
Government schemes database with eligibility criteria for matching with user profiles.
"""

SCHEMES = [
    {
        "name": "PM Kisan Samman Nidhi",
        "description": "Direct income support of Rs. 6,000 per year to farmer families",
        "eligibility_criteria": {
            "occupation": ["farmer"],
            "age": {"min": 18, "max": None},
            "location": "all",
            "gender": "all",
            "income_limit": 200000  # Annual income in INR
        },
        "benefits": "Rs. 6,000 per year in three equal installments",
        "documents_required": ["Aadhar Card", "Land Records", "Bank Account Details"],
        "application_process": "Apply online through PM Kisan portal or visit local agriculture office"
    },
    {
        "name": "Pradhan Mantri Awas Yojana (PMAY)",
        "description": "Housing subsidies for affordable housing for urban and rural poor",
        "eligibility_criteria": {
            "occupation": "all", 
            "age": {"min": 18, "max": None},
            "location": "all",
            "gender": "all",
            "income_limit": {
                "EWS": 300000,  # Economically Weaker Section
                "LIG": 600000,  # Low Income Group
                "MIG-1": 1200000,  # Middle Income Group 1
                "MIG-2": 1800000   # Middle Income Group 2
            }
        },
        "benefits": "Interest subsidy up to Rs. 2.67 lakh on home loans",
        "documents_required": ["Aadhar Card", "Income Certificate", "Bank Account Details"],
        "application_process": "Apply through PMAY portal or visit local municipal office"
    },
    {
        "name": "Pradhan Mantri Ujjwala Yojana",
        "description": "Free LPG connections to women from BPL households",
        "eligibility_criteria": {
            "occupation": "all",
            "age": {"min": 18, "max": None},
            "location": "all",
            "gender": ["female"],
            "income_limit": 100000
        },
        "benefits": "Free LPG connection with first refill and stove",
        "documents_required": ["Aadhar Card", "BPL Card", "Bank Account Details"],
        "application_process": "Apply at nearest LPG distributor"
    },
    {
        "name": "Sukanya Samriddhi Yojana",
        "description": "Small savings scheme for girl child education and marriage expenses",
        "eligibility_criteria": {
            "occupation": "all",
            "age": {"min": 0, "max": 10},
            "location": "all",
            "gender": ["female"],
            "income_limit": None
        },
        "benefits": "High interest rate (currently 7.6%) and tax benefits under Section 80C",
        "documents_required": ["Girl's Birth Certificate", "Parent/Guardian ID", "Address Proof"],
        "application_process": "Open account at post office or authorized banks"
    },
    {
        "name": "Atal Pension Yojana",
        "description": "Pension scheme for unorganized sector workers",
        "eligibility_criteria": {
            "occupation": "all",
            "age": {"min": 18, "max": 40},
            "location": "all",
            "gender": "all",
            "income_limit": None
        },
        "benefits": "Fixed pension between Rs. 1,000 to Rs. 5,000 per month after 60 years of age",
        "documents_required": ["Aadhar Card", "Bank Account Details", "Mobile Number"],
        "application_process": "Apply through bank branch"
    },
    {
        "name": "National Scholarship Portal",
        "description": "Single-window system for scholarship schemes for students",
        "eligibility_criteria": {
            "occupation": ["student"],
            "age": {"min": 0, "max": 35},
            "location": "all",
            "gender": "all",
            "income_limit": 800000
        },
        "benefits": "Financial assistance for education",
        "documents_required": ["Institution Verification", "Income Certificate", "Bank Account Details"],
        "application_process": "Apply online through National Scholarship Portal"
    },
    {
        "name": "Pradhan Mantri Vaya Vandana Yojana",
        "description": "Pension scheme for senior citizens",
        "eligibility_criteria": {
            "occupation": "all",
            "age": {"min": 60, "max": None},
            "location": "all",
            "gender": "all",
            "income_limit": None
        },
        "benefits": "Assured return of 7.40% per annum for 10 years",
        "documents_required": ["Age Proof", "Identity Proof", "Address Proof"],
        "application_process": "Apply through LIC branches"
    },
    {
        "name": "Ayushman Bharat - PMJAY",
        "description": "Health insurance coverage of Rs. 5 lakh per family per year",
        "eligibility_criteria": {
            "occupation": "all",
            "age": {"min": 0, "max": None},
            "location": "all",
            "gender": "all",
            "income_limit": None,
            "criteria": "Families identified based on deprivation criteria from SECC database"
        },
        "benefits": "Cashless and paperless access to healthcare services",
        "documents_required": ["Aadhar Card", "Ration Card/SECC Database Entry"],
        "application_process": "Check eligibility on PMJAY website or visit nearest Ayushman Bharat Kendra"
    }
]

def match_schemes(user_info):
    """
    Match user profile with eligible schemes.
    
    Args:
        user_info (dict): User profile containing age, gender, location, etc.
    
    Returns:
        list: List of matching schemes
    """
    matching_schemes = []
    
    for scheme in SCHEMES:
        eligible = True
        criteria = scheme["eligibility_criteria"]
        
        # Check age eligibility
        if "age" in user_info and user_info["age"] is not None and criteria["age"] is not None:
            user_age = int(user_info["age"])
            min_age = criteria["age"]["min"]
            max_age = criteria["age"]["max"]
            
            if min_age is not None and user_age < min_age:
                eligible = False
            if max_age is not None and user_age > max_age:
                eligible = False
        
        # Check gender eligibility
        if "gender" in user_info and user_info["gender"] is not None and criteria["gender"] != "all":
            if user_info["gender"].lower() not in criteria["gender"]:
                eligible = False
        
        # For now, we assume location is eligible for all schemes
        # In a real implementation, you would check specific location eligibility
        
        if eligible:
            matching_schemes.append(scheme)
    
    return matching_schemes 