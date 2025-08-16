import { DotLottieReact } from '@lottiefiles/dotlottie-react';

export function Loading() {
  return (
    <div className="flex items-center justify-center min-h-screen">
      <div className="text-center">
        <DotLottieReact
          src="https://lottie.host/5e07af5b-afa6-49be-9c5a-ec63f1655d11/SG0m9nxrKC.lottie"
          loop
          autoplay
          style={{ width: '200px', height: '200px' }}
        />
        <p className="mt-4 text-lg text-gray-600 dark:text-white">로딩 중...</p>
      </div>
    </div>
  );
}