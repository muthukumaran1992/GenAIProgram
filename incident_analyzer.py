incidents = [
    {
        "id": "INC001",
        "priority": "P1",
        "status": "Open",
        "sla_breached": True
    },
    {
        "id": "INC002",
        "priority": "P2",
        "status": "Resolved",
        "sla_breached": False
    },
    {
        "id": "INC003",
        "priority": "P3",
        "status": "Open",
        "sla_breached": True
    },
    {
        "id": "INC004",
        "priority": "P1",
        "status": "Resolved",
        "sla_breached": True
    }
]

def total_incidents(incidents):   
    return len(incidents)

def number_of_p1_incidents(incidents):
    cnt = 0
    for incident in incidents:
        if incident["priority"] == "P1":
            cnt += 1
    return cnt

def number_of_sla_breached_incidents(incidents):
    cnt = 0
    for incident in incidents:
        if incident["sla_breached"]:
            cnt += 1
    return cnt

def list_p1_p2_incidents(incidents):
    p1_p2_incidents = []
    for incident in incidents:
        if incident["priority"] in ["P1", "P2"]:
            p1_p2_incidents.append(incident["id"] + "-" + incident["priority"])
    return p1_p2_incidents

def project_status(incidents):
    status = "Green"
    cnt = 0
    for incident in incidents:
        if incident["sla_breached"] == True:
            cnt += 1
    if cnt > 2:       
        return "Red"
    elif cnt == 2:
        return "Amber"
    else:  
        return "Green"

    return status
    

print("total_incidents : ", total_incidents(incidents))
print("p1_incidents : ", number_of_p1_incidents(incidents))
print("sla_breached_incidents : ", number_of_sla_breached_incidents(incidents))
print("p1_p2_incidents : ", list_p1_p2_incidents(incidents))
print("project_status :", project_status(incidents))