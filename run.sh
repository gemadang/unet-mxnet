#sh
#!/bin/bash - x

#APT_PACKAGES="apt-utils ffmpeg libav-tools x264 x265"
#apt-install() {
#	export DEBIAN_FRONTEND=noninteractive
#	apt-get update -q
#	apt-get install -q -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" $APT_PACKAGES
#	return $?
#}

#install ffmpeg to container
#add-apt-repository -y ppa:jonathonf/ffmpeg-3 2>&1
#apt-install || exit 1

pip install mxnet-cu90
pip install matplotlib
pip install numpy 
pip install scikit-image
pip install awscli --upgrade --ignore-installed six

aws s3 sync s3://japan-roof-top-bucket /storage --no-sign-request
#ls -la /storage/Train_label
#ls -la /storage/Train_data

apt-get update && apt-get install -y openssh-server
mkdir /var/run/sshd
echo 'root:password' | chpasswd
sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

echo "export VISIBLE=now" >> /etc/profile

/usr/sbin/sshd -D

python model.py
