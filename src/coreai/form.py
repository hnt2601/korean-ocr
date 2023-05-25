driver_form = {
    "Types": [0, 0, 282, 128],
    "Serial number": [300, 100, 740, 145],
    "Full name": [300, 160, 740, 217],
    "No": [300, 210, 740, 250],
    "Address": [309, 250, 740, 355],
    "Valid from": [300, 365, 650, 400],
    "Renewal date": [300, 410, 650, 440],
    "Issuer": [0, 510, 747, 590],
    "Conditions": [300, 445, 740, 500],
}


residencial_form = {
    "Full name": [50, 157, 381, 220],
    "No": [50, 230, 410, 300],
    "Address": [50, 310, 430, 430],
    "Issuer": [200, 500, 553, 560],
    "Issue date": [300, 430, 540, 500],
}


foreign_form = {
    "Full name": [286, 222, 629, 277],
    "No": [355, 130, 599, 182],
    "Issuer": [429, 494, 742, 539],
    "Issue date": [530, 450, 739, 492],
}


mask = {
    "Types": [""],
    "Serial number": "",  # passport number
    "Full name": "",
    "No": "",  # personal number passport / residencial number
    "Address": "",
    "Valid from": "",
    "Renewal date": "",
    "Issuer": "",
    "Conditions": [""],
    "Issue date": "",
}
