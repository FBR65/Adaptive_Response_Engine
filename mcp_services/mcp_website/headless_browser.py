import logging
from playwright.sync_api import sync_playwright
import trafilatura

# Configure logging for this module
logger = logging.getLogger(__name__)


class HeadlessBrowserExtractor:
    """
    Uses a headless browser (via Playwright) to load a webpage,
    allowing JavaScript rendering, and then extracts the main text content
    using Trafilatura.
    """

    def __init__(self, default_wait_time: int = 5):
        """
        Initializes the extractor.

        Args:
            default_wait_time (int): Default seconds to wait for page rendering
                                     if not specified in extract_text.
        """
        self.default_wait_time = default_wait_time
        logger.info("HeadlessBrowserExtractor initialized with Playwright.")

    def extract_text(self, url: str, wait_time: int = None) -> str | None:
        """
        Extracts the main text content from a webpage using Playwright.

        Args:
            url (str): The URL to extract text from.
            wait_time (int, optional): Seconds to wait for page rendering.
                                      Defaults to self.default_wait_time.

        Returns:
            str | None: Main text content if successful, None if error occurred.
        """
        effective_wait_time = (
            wait_time if wait_time is not None else self.default_wait_time
        )

        logger.info(f"Extracting text from {url} (wait_time={effective_wait_time}s)")

        try:
            with sync_playwright() as p:
                # Launch browser in headless mode
                browser = p.chromium.launch(headless=True)

                # Create a new page with custom user agent
                page = browser.new_page(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                )

                logger.debug(f"Navigating to {url}")
                page.goto(url, wait_until="domcontentloaded")

                logger.debug(
                    f"Waiting {effective_wait_time} seconds for potential JS rendering..."
                )
                page.wait_for_timeout(
                    effective_wait_time * 1000
                )  # Playwright uses milliseconds

                page_content = page.content()
                browser.close()

                if not page_content:
                    logger.warning(f"Could not retrieve page content from {url}")
                    return None

                logger.debug(
                    f"Page content retrieved ({len(page_content)} bytes). Extracting text with Trafilatura."
                )

                # Extract using Trafilatura
                extracted_text = trafilatura.extract(
                    page_content,
                    url=url,
                    output_format="txt",
                    include_comments=False,
                    favor_recall=True,
                )

                if extracted_text:
                    logger.info(
                        f"Text successfully extracted from {url} using Trafilatura."
                    )
                    # Clean whitespace
                    return " ".join(extracted_text.split())
                else:
                    logger.warning(
                        f"Trafilatura found no main text on {url} after rendering. Trying fallback."
                    )
                    # Fallback: Try body text using Playwright
                    try:
                        # Reopen page for fallback (browser was closed)
                        browser = p.chromium.launch(headless=True)
                        page = browser.new_page()
                        page.goto(url, wait_until="domcontentloaded")
                        page.wait_for_timeout(effective_wait_time * 1000)

                        body_text = page.inner_text("body")
                        browser.close()

                        if body_text:
                            logger.info("Using fallback: Text from body element.")
                            return " ".join(body_text.split())
                        else:
                            logger.info(
                                f"Fallback failed: Body text is empty for {url}."
                            )
                            return ""
                    except Exception as body_e:
                        logger.error(
                            f"Error during fallback body text retrieval from {url}: {body_e}"
                        )
                        return ""

        except Exception as e:
            logger.error(
                f"Playwright error during extraction from {url} ({type(e).__name__}): {e}"
            )
            return None

    def extract_text_from_multiple_urls(
        self, urls: list[str], wait_time: int = None
    ) -> dict[str, str | None]:
        """
        Extracts text from multiple URLs.

        Args:
            urls (list[str]): List of URLs to extract text from.
            wait_time (int, optional): Wait time for each URL.

        Returns:
            dict[str, str | None]: Dictionary mapping URLs to extracted text.
        """
        results = {}
        for url in urls:
            try:
                results[url] = self.extract_text(url, wait_time)
            except Exception as e:
                logger.error(f"Failed to extract text from {url}: {e}")
                results[url] = None
        return results
