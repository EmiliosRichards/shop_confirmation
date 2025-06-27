import os
import logging
from typing import Optional
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

logger = logging.getLogger(__name__)

def send_slack_notification(token: str, channel_id: str, message: str, file_path: Optional[str] = None):
    """
    Sends a message to a Slack channel and optionally uploads a file.

    Args:
        token (str): Your Slack Bot User OAuth Token.
        channel_id (str): The ID of the channel to post to.
        message (str): The text message to send.
        file_path (str, optional): The path to the file to upload. Defaults to None.
    """
    if not token or not channel_id:
        logger.warning("Slack token or channel ID is not configured. Skipping notification.")
        return

    client = WebClient(token=token)

    try:
        # Send the text message
        result = client.chat_postMessage(
            channel=channel_id,
            text=message
        )
        if result and result.get("ts"):
            logger.info(f"Slack message sent successfully: {result['ts']}")
        else:
            logger.warning("Slack message post did not return expected timestamp.")

        # If a file path is provided, upload the file
        if file_path and os.path.exists(file_path):
            file_upload_result = client.files_upload_v2(
                channel=channel_id,
                file=file_path,
                initial_comment=f"Attached is the report: {os.path.basename(file_path)}",
                title=os.path.basename(file_path)
            )
            if file_upload_result and file_upload_result.get("file"):
                if file_upload_result:
                    file_info = file_upload_result.get("file")
                    if isinstance(file_info, dict) and file_info.get("name"):
                        logger.info(f"File uploaded successfully to Slack: {file_info['name']}")
                    else:
                        logger.warning("Slack file upload response did not contain expected file information.")
                else:
                    logger.warning("Slack file upload did not return a result.")
            else:
                logger.warning("Slack file upload did not return expected file information.")
        elif file_path:
            logger.warning(f"File path provided for Slack upload, but file not found at: {file_path}")

    except SlackApiError as e:
        if e.response and "error" in e.response:
            logger.error(f"Error sending Slack notification: {e.response['error']}", exc_info=True)
        else:
            logger.error(f"An unexpected Slack API error occurred: {e}", exc_info=True)
    except FileNotFoundError:
        logger.error(f"Error uploading file to Slack: File not found at {file_path}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred during Slack notification: {e}", exc_info=True)