# Set up the docker repository
sudo apt-get update
sudo apt-get install apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"

# Install Docker CE
sudo apt-get update
sudo apt-get install docker-ce

# Pull Machine learning Python image 
docker pull frolvlad/alpine-python-machinelearning

#Execute script
docker run --rm frolvlad/alpine-python-machinelearning python3 -c 'import numpy; print(numpy.arange(3))'

