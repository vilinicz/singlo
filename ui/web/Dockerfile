# ui/web/Dockerfile
FROM node:18-alpine

WORKDIR /app
ENV CI=true \
    VITE_PORT=3000 \
    HOST=0.0.0.0

COPY package.json ./
RUN npm i

COPY . .
EXPOSE 3000
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "3000"]
