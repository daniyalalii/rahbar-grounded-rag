FROM nginxinc/nginx-unprivileged:1.29-alpine

# Pull in Alpine security fixes published after the base image was built (Trivy findings in CI).
USER root
RUN apk upgrade --no-cache
USER 101

COPY nginx.conf /etc/nginx/nginx.conf
COPY index.html /usr/share/nginx/html/index.html

EXPOSE 8080