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

def project_status(incidents):
    cnt = 0

    for incident in incidents:
        if incident["sla_breached"]:
            cnt += 1

    if cnt > 2:       
        return "Red"
    elif cnt == 2:
        return "Amber"
    else:  
        return "Green"

def incident_summary(incidents):
    tot_cnt = 0
    p1_cnt = 0
    p2_cnt = 0
    open_cnt = 0 
    resolved_cnt = 0
    sla_cnt = 0

    tot_cnt = len(incidents)
    for incident in incidents:
        if incident["priority"] == "P1":
            p1_cnt += 1
        if incident["priority"] == "P2":
            p2_cnt += 1
        if incident["sla_breached"]:
            sla_cnt += 1
        if incident["status"] == "Open":
            open_cnt += 1
        elif incident["status"] == "Resolved": 
              resolved_cnt += 1  

    return {
        "Total": tot_cnt,
        "P1": p1_cnt,
        "P2": p2_cnt,
        "sla_breaches": sla_cnt,
        "open": open_cnt,
        "resolved": resolved_cnt,
        "ProjectStatus":   project_status(incidents)    
    }   

summary = incident_summary(incidents)

print(summary)
