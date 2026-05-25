"""
camera_test.py
==============
카메라 권한 팝업을 띄우고 웹캠이 정상 작동하는지 확인하는 단독 테스트.

macOS에서 터미널이 '카메라 접근 허용' 목록에 등록되려면
메인 스레드에서 카메라를 한 번 열어야 한다. 이 스크립트가 그 역할.

사용:
    본인 터미널에서 직접 실행 (중요!)
    .venv/bin/python camera_test.py

조작:
    q : 종료
"""

import cv2

print("카메라를 엽니다... macOS 권한 팝업이 뜨면 '허용'을 누르세요.")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[실패] 카메라를 열 수 없습니다.")
    print("  → 시스템 설정 > 개인정보 보호 및 보안 > 카메라 에서")
    print("    '터미널'을 켠 뒤 터미널을 재시작하고 다시 실행하세요.")
    raise SystemExit(1)

print("[성공] 카메라가 열렸습니다! 창이 뜨면 q를 눌러 종료하세요.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[경고] 프레임을 읽지 못했습니다.")
        break
    cv2.imshow("Camera Test (press q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("테스트 완료. 카메라가 정상 작동합니다!")
