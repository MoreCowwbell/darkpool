#!/usr/bin/env python
"""
Discord webhook posting module for darkpool analysis outputs.

Posts images and tables to Discord webhook for real-time notifications.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

import requests


def post_to_discord(
    webhook_url: str,
    file_paths: Union[Path, List[Path]],
    content: Optional[str] = None,
    username: str = "Darkpool Bot",
) -> bool:
    """
    Post one or more files to a Discord webhook.

    Args:
        webhook_url: Discord webhook URL
        file_paths: Single file path or list of file paths to upload
        content: Optional message text to include with the files
        username: Bot username to display in Discord

    Returns:
        True if successful, False otherwise
    """
    if not webhook_url:
        logging.warning("Discord webhook URL not configured, skipping post")
        return False

    # Normalize to list
    if isinstance(file_paths, Path):
        file_paths = [file_paths]

    # Filter to existing files
    existing_files = [p for p in file_paths if p.exists()]
    if not existing_files:
        logging.warning("No files found to post to Discord")
        return False

    try:
        # Discord allows up to 10 files per message
        # Split into batches if needed
        batch_size = 10
        success = True

        for i in range(0, len(existing_files), batch_size):
            batch = existing_files[i:i + batch_size]

            # Prepare multipart form data
            files = {}
            for idx, file_path in enumerate(batch):
                file_key = f"file{idx}"
                files[file_key] = (
                    file_path.name,
                    open(file_path, "rb"),
                    _get_content_type(file_path),
                )

            # Payload with optional content
            payload = {"username": username}
            if content and i == 0:  # Only include content with first batch
                payload["content"] = content

            response = requests.post(
                webhook_url,
                data=payload,
                files=files,
                timeout=60,
            )

            # Close file handles
            for f in files.values():
                f[1].close()

            if response.status_code not in (200, 204):
                logging.error(
                    "Discord webhook failed: %s - %s",
                    response.status_code,
                    response.text[:200] if response.text else "No response",
                )
                success = False
            else:
                logging.info(
                    "Posted %d file(s) to Discord: %s",
                    len(batch),
                    ", ".join(p.name for p in batch),
                )

        return success

    except requests.exceptions.Timeout:
        logging.error("Discord webhook timed out")
        return False
    except requests.exceptions.RequestException as e:
        logging.error("Discord webhook error: %s", e)
        return False
    except Exception as e:
        logging.error("Unexpected error posting to Discord: %s", e)
        return False


def post_image_to_discord(
    webhook_url: str,
    image_path: Path,
    title: Optional[str] = None,
    description: Optional[str] = None,
) -> bool:
    """
    Post a single image to Discord with optional title and description.

    Args:
        webhook_url: Discord webhook URL
        image_path: Path to the image file
        title: Optional title to include as message content
        description: Optional description to include

    Returns:
        True if successful, False otherwise
    """
    content = None
    if title or description:
        parts = []
        if title:
            parts.append(f"**{title}**")
        if description:
            parts.append(description)
        content = "\n".join(parts)

    return post_to_discord(webhook_url, image_path, content=content)


def post_images_batch(
    webhook_url: str,
    image_paths: List[Path],
    title: Optional[str] = None,
) -> bool:
    """
    Post multiple images to Discord in a single message (up to 10).

    Args:
        webhook_url: Discord webhook URL
        image_paths: List of image file paths
        title: Optional title for the batch

    Returns:
        True if successful, False otherwise
    """
    return post_to_discord(webhook_url, image_paths, content=title)


def _get_content_type(file_path: Path) -> str:
    """Get MIME content type for a file based on extension."""
    suffix = file_path.suffix.lower()
    content_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".html": "text/html",
        ".csv": "text/csv",
        ".json": "application/json",
        ".pdf": "application/pdf",
    }
    return content_types.get(suffix, "application/octet-stream")


# Convenience functions for specific output types
def post_metrics_table(webhook_url: str, table_path: Path, date_str: str) -> bool:
    """Post daily metrics table to Discord."""
    return post_image_to_discord(
        webhook_url,
        table_path,
        title=f"Daily Metrics Table - {date_str}",
    )


def post_metrics_plot(webhook_url: str, plot_path: Path, ticker: str) -> bool:
    """Post metrics plot to Discord."""
    return post_image_to_discord(
        webhook_url,
        plot_path,
        title=f"Metrics Plot - {ticker}",
    )


def post_price_chart(webhook_url: str, chart_path: Path, ticker: str) -> bool:
    """Post price chart to Discord."""
    return post_image_to_discord(
        webhook_url,
        chart_path,
        title=f"Price Chart - {ticker}",
    )


def post_summary_dashboard(webhook_url: str, dashboard_path: Path, date_str: str) -> bool:
    """Post summary dashboard to Discord."""
    return post_image_to_discord(
        webhook_url,
        dashboard_path,
        title=f"Sector Summary - {date_str}",
    )


def post_all_tickers_dashboard(webhook_url: str, dashboard_path: Path, date_str: str) -> bool:
    """Post all-tickers dashboard to Discord."""
    return post_image_to_discord(
        webhook_url,
        dashboard_path,
        title=f"All Tickers Dashboard - {date_str}",
    )


def post_combination_plot(webhook_url: str, plot_path: Path, date_str: str) -> bool:
    """Post combination plot to Discord."""
    return post_image_to_discord(
        webhook_url,
        plot_path,
        title=f"Combination Plot - {date_str}",
    )


def post_circos_plot(webhook_url: str, plot_path: Path, date_str: str) -> bool:
    """Post circos chord diagram to Discord."""
    return post_image_to_discord(
        webhook_url,
        plot_path,
        title=f"Circos Money Flow - {date_str}",
    )


def post_wtd_vwbr_plot(webhook_url: str, plot_path: Path, ticker: str) -> bool:
    """Post WTD VWBR plot to Discord."""
    return post_image_to_discord(
        webhook_url,
        plot_path,
        title=f"WTD VWBR - {ticker}",
    )
