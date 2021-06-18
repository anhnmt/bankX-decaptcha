import requests, time

url = 'https://vcbdigibank.vietcombank.com.vn/w1/get-captcha/a7dac5a5-a21e-52ce-2eed-f2c44a034279'
for i in range(1000):
	r = requests.get(url, allow_redirects=True)
	open('original/%s.png'%i, 'wb').write(r.content)
	print(i)
	time.sleep(1)
