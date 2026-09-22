from incidents import incidents


def create_summary(incident_list):
    return {
        "Total": len(incident_list),
        "P1": sum(incident["priority"] == "P1" for incident in incident_list),
        "P2": sum(incident["priority"] == "P2" for incident in incident_list),
        "sla_breaches": sum(
            incident["sla_breached"] for incident in incident_list
        ),
        "open": sum(incident["status"] == "Open" for incident in incident_list),
        "resolved": sum(
            incident["status"] == "Resolved" for incident in incident_list
        ),
    }


def print_priority_incidents(incident_list):
    print("\nP1/P2 incidents:")
    for incident in incident_list:
        if incident["priority"] in ("P1", "P2"):
            print(
                f"{incident['id']}: {incident['priority']} - "
                f"{incident['application']} - {incident['status']}"
            )

def print_affected_apps(incident_list):
    affected_apps = set()
    for incident in incident_list:
      if incident["priority"] in ("P1", "P2"):
            affected_apps.add(incident["application"])
    print("\nAffected Applications with SLA breaches:")
    for app in affected_apps:
        print(app)

def print_related_bugs(incident_list):
    related_bugs = set()
    for incident in incident_list:
     if incident["priority"] in ("P1", "P2"):
        related_bugs.update(incident["related_bugs"])
    print("\nRelated Bugs:")
    for bug in related_bugs:
        print(bug)

def print_deployments(incident_list):
    deployments = set()
    for incident in incident_list:
     if incident["priority"] in ("P1", "P2"):
        deployments.update(incident["deployments"])
    print("\nDeployments:")
    for deployment in deployments:
        print(deployment)

def print_comments(incident_list):
    print("\nComments for P1/P2 incidents:")
    for incident in incident_list:
        if incident["priority"] in ("P1", "P2"):
             for comment in incident["comments"]:
                print(f"- {comment}") 

def project_status(incident_list):
    cnt = 0

    for incident in incident_list:
        if incident["sla_breached"]:
            cnt += 1

    if cnt > 2:       
        return "Red"
    elif cnt == 2:
        return "Amber"
    else:  
        return "Green"


summary = create_summary(incidents)
summary["project_status"] = project_status(incidents)

for key, value in summary.items():
    print(f"{key}: {value}")

print_priority_incidents(incidents)
print_affected_apps(incidents)
print_related_bugs(incidents)
print_deployments(incidents)
print_comments(incidents)
