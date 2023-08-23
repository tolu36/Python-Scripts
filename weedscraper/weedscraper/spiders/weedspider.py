import scrapy


class WeedspiderSpider(scrapy.Spider):
    name = "weedspider"
    allowed_domains = ["thehunnypot.com"]
    start_urls = [
        "https://dutchie.com/embedded-menu/the-hunny-pot-cannabis-co1/products/flower"
    ]

    def parse(self, response):
        pass
