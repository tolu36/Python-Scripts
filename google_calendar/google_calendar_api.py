# %%
# delete token to refresh this process it will make the user login again
from __future__ import print_function

import datetime as dt
import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pandas_market_calendars as cal

# If modifying these scopes, delete the file token.json.
SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/tasks",
]


class GoogleTask:
    def __init__(self, SCOPES, switch_email=False, max_results=10) -> None:
        self.SCOPES = SCOPES
        self.switch_email = switch_email
        self.maxResults = max_results

    def connect_to_api(self):
        self.creds = None
        # The file token.json stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first
        # time.

        if os.path.exists("token.json"):
            self.creds = Credentials.from_authorized_user_file(
                "token.json", self.SCOPES
            )
        # If there are no (valid) credentials available, let the user log in.
        if not self.creds or not self.creds.valid or self.switch_email:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    "credentials.json", self.SCOPES
                )
                self.creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open("token.json", "w") as token:
                token.write(self.creds.to_json())

    def get_task(self):
        try:
            self.service = build("tasks", "v1", credentials=self.creds)

            # Call the Tasks API
            self.results = (
                self.service.tasklists().list(maxResults=self.maxResults).execute()
            )
            items = self.results.get("items", [])

            if not items:
                print("No task lists found.")
                return

            print("Task lists:")
            self.task_dict = {}
            for item in items:
                print("{0} ({1})".format(item["title"], item["id"]))
                item_list = self.service.tasks().list(tasklist=item["id"]).execute()
                task_list = []
                for task in item_list.get("items", []):
                    task_list.append(
                        {
                            "title": task.get("title"),
                            "due": task.get("due"),
                            "task_list_id": item["id"],
                        }
                    )
                self.task_dict[item["title"]] = task_list

        except HttpError as err:
            print(err)

    def create_task(self, task_list_name, task_body):
        try:
            self.service = build("tasks", "v1", credentials=self.creds)

            # Call the Tasks API
            self.results = (
                self.service.tasklists().list(maxResults=self.maxResults).execute()
            )
            items = self.results.get("items", [])
            if not items:
                print("No task lists found.")
                return
            for item in items:
                if item["title"] == task_list_name:
                    self.service.tasks().insert(
                        tasklist=item["id"], body=task_body
                    ).execute()
                else:
                    print(f"No task list by the name {task_list_name} found.")
                    return
        except HttpError as err:
            print(err)


# %%
task_list = GoogleTask(SCOPES)
task_list.connect_to_api()

tsx = cal.get_calendar("TSX")
early = tsx.schedule(start_date="2024-01-01", end_date="2024-12-31")
date_list = cal.date_range(early, frequency="1D")

# %%
for mon in sorted(list(date_list.strftime("%m").unique())):
    tmp = sorted(date_list[date_list.strftime("%m") == mon])[-2]
    body = {
        "status": "needsAction",
        "kind": "tasks#task",
        "updated": dt.datetime.now().isoformat() + "Z",
        "parent": "Mortgage interest checker",
        "links": [],
        "title": f"Check what the interest accrued is for the month of {tmp.strftime('%B')}.",
        "due": tmp.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z",
    }
    task_list.create_task("My Tasks", body)


# %%
# def main(switch_email=False, get_task=False):
#     """Shows basic usage of the Google Calendar API.
#     Prints the start and name of the next 10 events on the user's calendar.
#     """
#     creds = None
#     # The file token.json stores the user's access and refresh tokens, and is
#     # created automatically when the authorization flow completes for the first
#     # time.
#     if os.path.exists("token.json"):
#         creds = Credentials.from_authorized_user_file("token.json", SCOPES)
#     # If there are no (valid) credentials available, let the user log in.
#     if not creds or not creds.valid or switch_email:
#         if creds and creds.expired and creds.refresh_token:
#             creds.refresh(Request())
#         else:
#             flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
#             creds = flow.run_local_server(port=0)
#         # Save the credentials for the next run
#         with open("token.json", "w") as token:
#             token.write(creds.to_json())

#     try:
#         service = build("calendar", "v3", credentials=creds)

#         # Call the Calendar API
#         now = dt.datetime.utcnow().isoformat() + "Z"  # 'Z' indicates UTC time
#         print("Getting the upcoming 10 events")
#         events_result = (
#             service.events()
#             .list(
#                 calendarId="primary",
#                 timeMin=now,
#                 maxResults=10,
#                 singleEvents=True,
#                 orderBy="startTime",
#             )
#             .execute()
#         )
#         events = events_result.get("items", [])
#         if not events:
#             print("No upcoming events found.")
#             return

#         # Prints the start and name of the next 10 events
#         for event in events:
#             start = event["start"].get("dateTime", event["start"].get("date"))
#             print(start, event["summary"])

#     except HttpError as error:
#         print("An error occurred: %s" % error)

#     if get_task:
#         try:
#             service = build("tasks", "v1", credentials=creds)

#             # Call the Tasks API
#             results = service.tasklists().list(maxResults=10).execute()
#             items = results.get("items", [])

#             if not items:
#                 print("No task lists found.")
#                 return

#             print("Task lists:")
#             task_dict = {}
#             for item in items:
#                 print("{0} ({1})".format(item["title"], item["id"]))
#                 item_list = service.tasks().list(tasklist=item["id"]).execute()
#                 task_list = []
#                 for task in item_list.get("items", []):
#                     task_list.append([task.get("title"), task.get("due"), item["id"]])
#                 task_dict[item["title"]] = task_list

#         except HttpError as err:
#             print(err)
#     return events, task_dict


# if __name__ == "__main__":
#     events, tasks = main(get_task=True)

# # %%
