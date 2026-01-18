# Daily Report 스케줄링 설정 가이드

매일 아침 08:00 KST에 자산배분/Head-Tail 리포트를 이메일로 전송하기 위한 설정 방법입니다.

---

## 🚀 GitHub Actions (권장)

GitHub Actions를 사용하면 서버 없이도 자동화할 수 있습니다.

### Variables 설정 (민감하지 않은 값)

GitHub Repository → Settings → Secrets and variables → Actions → Variables → New repository variable

| Variable Name | 설명 | 예시 |
|---------------|------|------|
| `SMTP_HOST` | SMTP 서버 주소 | `smtp.gmail.com` |
| `SMTP_PORT` | SMTP 포트 | `587` |
| `SMTP_USER` | SMTP 사용자 (이메일) | `your-email@gmail.com` |
| `SMTP_STARTTLS` | TLS 사용 여부 | `true` |
| `MAIL_FROM` | 발신자 이메일 | `your-email@gmail.com` |
| `MAIL_TO` | 수신자 (쉼표 구분) | `a@example.com,b@example.com` |

### Secrets 설정 (진짜 민감한 값)

GitHub Repository → Settings → Secrets and variables → Actions → Secrets → New repository secret

| Secret Name | 설명 | 예시 |
|-------------|------|------|
| `SMTP_PASS` | SMTP 비밀번호 (앱 비밀번호) | `xxxx xxxx xxxx xxxx` |

### 수동 실행 (즉시 실행)

1. GitHub Repository → Actions 탭
2. 실행할 워크플로우 선택 (`daily-rank-email`, `daily-asset-allocation-email`, `daily-reports-email`)
3. "Run workflow" 버튼 클릭

### 자동 실행

- 매일 **08:00 KST** (UTC 23:00)에 자동 실행됩니다.
- 워크플로우 파일:  
  - `daily_rank_email.yml`
  - `daily_asset_allocation_email.yml`
  - `daily_reports_email.yml` (중복 방지용으로 기본 비활성화)

---

## 추가 안내

이 문서는 GitHub Actions 기반 운영을 전제로 정리했습니다.  
과거에 사용하던 로컬 스크립트(`daily_report.py`) 및 Cron/launchd 섹션은 정리되어 제거되었습니다.
