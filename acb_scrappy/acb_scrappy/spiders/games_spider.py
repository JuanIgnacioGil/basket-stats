import scrapy


class GamesSpider(scrapy.Spider):
    name = "games"

    def start_requests(self):
        url = 'http://www.acb.com'
        game = getattr(self, 'game', None)

        if game is not None:
            url = '{}/fichas/{}.php'.format(url, game)

        yield scrapy.Request(url, self.parse)

    def parse(self, response):
        for quote in response.css('div.quote'):
            yield {
                'text': quote.css('span.text::text').extract_first(),
                'author': quote.css('small.author::text').extract_first(),
            }

        next_page = response.css('li.next a::attr(href)').extract_first()
        if next_page is not None:
            yield response.follow(next_page, self.parse)